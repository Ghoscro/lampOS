"""Fire-hazard detection: request shape, gate/escalation, confirm and alert logic."""

import json
import logging

import numpy as np
import pytest

from hal import config
from hal.drivers.sensing.crypto import DLCrypto
from hal.drivers.sensing.perceptions.processors import fire_hazard
from hal.drivers.sensing.perceptions.processors.fire_hazard import RemoteFireHazardDetector

FRAME = np.zeros((64, 64, 3), dtype=np.uint8)
FIRE = "a single flame"
SMOKE = "a single cloud of smoke"
FLAMMABLE = "a single piece of wooden furniture"
SAFE = "a single fireplace"


class _Resp:
    def __init__(self, status: int, body=None):
        self.status_code = status
        self._body = body if body is not None else {"detections": []}
        self.text = json.dumps(self._body)
        self.content = self.text.encode()

    def json(self):
        return self._body


class _FakePost:
    """Scripted stand-in for requests.post; records each posted JSON body."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.bodies: list[dict] = []

    def __call__(self, url, data=None, headers=None, timeout=None):
        self.bodies.append(json.loads(data))
        item = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        if isinstance(item, BaseException):
            raise item
        return item


def _det(name: str, xywh=(0.5, 0.5, 0.2, 0.2), conf=0.9) -> dict:
    return {"class_name": name, "confidence": conf, "xywh": list(xywh)}


@pytest.fixture
def plaintext(monkeypatch):
    monkeypatch.setattr(DLCrypto, "http_session", lambda self: None)


@pytest.fixture
def post(monkeypatch, plaintext):
    fake = _FakePost(_Resp(200))
    monkeypatch.setattr(fire_hazard.requests, "post", fake)
    return fake


def _detector() -> RemoteFireHazardDetector:
    return RemoteFireHazardDetector(base_url="https://dl.example/detect/owlv2", api_key="k")


# --- request shape (Task 3.2) -------------------------------------------------

def test_sends_preset_not_classes(post, monkeypatch):
    monkeypatch.setattr(config, "FIRE_HAZARD_PRESET", "fire_hazard_v1")
    _detector().detect(FRAME)
    assert len(post.bodies) == 1
    assert post.bodies[0]["preset"] == "fire_hazard_v1"
    assert "classes" not in post.bodies[0]
    assert post.bodies[0]["image_b64"]


def test_400_falls_back_to_classes_same_call(post, monkeypatch):
    monkeypatch.setattr(config, "FIRE_HAZARD_PRESET", "fire_hazard_v1")
    post.responses = [_Resp(400, {"detail": "unknown preset"}), _Resp(200, {"detections": [_det(FIRE)]})]
    hazards = _detector().detect(FRAME)
    assert len(post.bodies) == 2
    assert "preset" in post.bodies[0]
    assert "preset" not in post.bodies[1]
    assert len(post.bodies[1]["classes"]) == 46
    assert post.bodies[1]["image_b64"] == post.bodies[0]["image_b64"]
    assert [h.type.value for h in hazards] == ["unsure_fire"]


def test_empty_preset_config_sends_classes(post, monkeypatch):
    monkeypatch.setattr(config, "FIRE_HAZARD_PRESET", "")
    _detector().detect(FRAME)
    assert len(post.bodies) == 1
    assert "preset" not in post.bodies[0]
    assert len(post.bodies[0]["classes"]) == 46


def test_unknown_classes_warning_logged_once(post, monkeypatch, caplog):
    monkeypatch.setattr(config, "FIRE_HAZARD_PRESET", "fire_hazard_v1")
    monkeypatch.setattr(fire_hazard, "_warned_unknown_preset_classes", False)
    post.responses = [_Resp(200, {"detections": [_det("person"), _det("chair")]})]
    det = _detector()
    with caplog.at_level(logging.WARNING):
        det.detect(FRAME)
        det.detect(FRAME)
    hits = [r for r in caplog.records if "preset response has no known classes" in r.getMessage()]
    assert len(hits) == 1


def test_known_classes_do_not_warn(post, monkeypatch, caplog):
    monkeypatch.setattr(config, "FIRE_HAZARD_PRESET", "fire_hazard_v1")
    monkeypatch.setattr(fire_hazard, "_warned_unknown_preset_classes", False)
    post.responses = [_Resp(200, {"detections": [_det(FIRE), _det("person")]})]
    with caplog.at_level(logging.WARNING):
        _detector().detect(FRAME)
    assert not [r for r in caplog.records if "no known classes" in r.getMessage()]


# --- gate + escalation (Task 3.3) ----------------------------------------------

from hal.drivers.sensing.perceptions.processors.fire_gate import GateResult, GateVerdict  # noqa: E402
from hal.drivers.sensing.perceptions.processors.fire_hazard import FireHazardPerception  # noqa: E402
from hal.drivers.sensing.perceptions.processors.models import FireHazard, FireHazardEnum  # noqa: E402

T0 = 1000.0
STEP = 6.0  # effective check cadence: 2 s sensing tick + 5 s interval gate (D6)


def _hz(kind: FireHazardEnum) -> FireHazard:
    return FireHazard(type=kind, bbox=np.array([10, 10, 30, 30], dtype=np.float32))


FIRE_HZ = [_hz(FireHazardEnum.UNSURE_FIRE)]


class _Clock:
    def __init__(self, t: float = T0):
        self.t = t

    def __call__(self) -> float:
        return self.t


class _FakeGate:
    def __init__(self, *verdicts: GateVerdict):
        self.verdicts = list(verdicts) or [GateVerdict.MISS]
        self.runs: list[object] = []
        self.closed = False

    @property
    def state(self) -> str:
        return "ready"

    def run(self, frame) -> GateResult:
        self.runs.append(frame)
        v = self.verdicts.pop(0) if len(self.verdicts) > 1 else self.verdicts[0]
        return GateResult(v, 0.5 if v is GateVerdict.HIT else 0.0, "fire" if v is GateVerdict.HIT else None, 1.0)

    def close(self) -> None:
        self.closed = True


class _FakeDetector:
    def __init__(self, *results):
        self.results = list(results) or [[]]
        self.frames: list[object] = []

    def detect(self, frame):
        self.frames.append(frame)
        return self.results.pop(0) if len(self.results) > 1 else self.results[0]


class _Events:
    def __init__(self, clock: _Clock):
        self.clock = clock
        self.sent: list[tuple[float, str]] = []

    def __call__(self, event, message, kind, snapshots, extra):
        self.sent.append((self.clock.t, message))


def _perception(gate, detector, clock=None):
    clock = clock or _Clock()
    events = _Events(clock)
    p = FireHazardPerception(None, events, gate=gate, detector=detector, clock=clock)
    return p, clock, events


def _run_checks(p, clock, n, frame=FRAME):
    """n due checks, STEP seconds apart, starting at the clock's current time."""
    for i in range(n):
        if i:
            clock.t += STEP
        p._check_impl(frame)


def test_gate_miss_skips_owlv2():
    gate, det = _FakeGate(GateVerdict.MISS), _FakeDetector()
    p, clock, _ = _perception(gate, det)
    _run_checks(p, clock, 5)
    assert det.frames == []
    assert len(gate.runs) == 5


def test_gate_hit_calls_owlv2_same_frame():
    gate, det = _FakeGate(GateVerdict.HIT), _FakeDetector([])
    p, clock, _ = _perception(gate, det)
    frame = np.ones((32, 32, 3), dtype=np.uint8)
    p._check_impl(frame)
    assert det.frames == [frame] and det.frames[0] is frame


def test_escalation_holds_while_owlv2_sees_fire():
    gate, det = _FakeGate(GateVerdict.HIT, GateVerdict.MISS), _FakeDetector(FIRE_HZ)
    p, clock, _ = _perception(gate, det)
    _run_checks(p, clock, 5)
    assert len(det.frames) == 5
    assert len(gate.runs) == 1


def test_escalation_ends_10s_after_last_sighting():
    # Fire seen at t=0 only: escalated until t=10, so t=6 still calls OWLv2 and
    # t=12 is back to the gate (MISS, no OWLv2 call).
    gate = _FakeGate(GateVerdict.HIT, GateVerdict.MISS)
    det = _FakeDetector(FIRE_HZ, [], [])
    p, clock, _ = _perception(gate, det)
    _run_checks(p, clock, 4)  # t = 0, 6, 12, 18
    assert len(det.frames) == 2
    assert len(gate.runs) == 3


def test_escalation_ends_10s_after_start_without_sighting():
    gate = _FakeGate(GateVerdict.HIT, GateVerdict.MISS)
    det = _FakeDetector([])
    p, clock, _ = _perception(gate, det)
    _run_checks(p, clock, 3)  # t = 0 (hit), 6 (escalated), 12 (gate miss)
    assert len(det.frames) == 2
    assert len(gate.runs) == 2


def test_gate_unavailable_is_fail_open():
    gate, det = _FakeGate(GateVerdict.UNAVAILABLE), _FakeDetector([])
    p, clock, _ = _perception(gate, det)
    _run_checks(p, clock, 5)
    assert len(det.frames) == 5


def test_need_owlv2_reasons():
    p, clock, _ = _perception(_FakeGate(GateVerdict.MISS, GateVerdict.UNAVAILABLE, GateVerdict.HIT), _FakeDetector())
    assert p._need_owlv2(T0, FRAME) == (False, "gate_miss")
    assert p._need_owlv2(T0, FRAME) == (True, "gate_unavailable")
    assert p._need_owlv2(T0, FRAME) == (True, "gate_hit")
    assert p._need_owlv2(T0 + 9.9, FRAME) == (True, "escalated")


def test_timeline_alert_latency_matches_today():
    sent_at = {}
    for verdict in (GateVerdict.HIT, GateVerdict.UNAVAILABLE):
        p, clock, events = _perception(_FakeGate(verdict, GateVerdict.MISS), _FakeDetector(FIRE_HZ))
        _run_checks(p, clock, 3)  # fire at 0 / 6 / 12
        assert len(events.sent) == 1
        sent_at[verdict] = events.sent[0][0] - T0
    assert sent_at[GateVerdict.HIT] == sent_at[GateVerdict.UNAVAILABLE] == 12.0


def test_to_dict_and_cleanup_report_gate():
    gate = _FakeGate(GateVerdict.HIT)
    p, clock, _ = _perception(gate, _FakeDetector([]))
    p._check_impl(FRAME)
    d = p.to_dict()
    assert d["gate_state"] == "ready" and d["escalated"] is True
    p.cleanup()
    assert gate.closed
