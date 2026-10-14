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
