"""Parity check: old single OWLv2 graph vs the text + image split, on real images.

Runs both on ONNX Runtime CPU with the 46 HAL fire-hazard prompts and compares raw
outputs, post-processed detections (threshold 0.1 + NMS, the server's path) and what
HAL would bucket (conf >= 0.3). Exits 1 on any breach of the tolerances below.

    python scripts/owlv2_split_parity.py --old owlv2_raw.onnx \
        --text owlv2_text_v1.onnx --image owlv2_image_v1.onnx [--images 'dir/*.jpg']
"""

import argparse
import glob
import json
import sys
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import cv2
import numpy as np
import onnxruntime as ort
from transformers import Owlv2Processor

from core.perception.object.predictors.onnx.base import ONNXObjectDetector
from core.utils.detection import owlv2_preprocess, unowlv2_boxes

HF_MODEL_ID = "google/owlv2-large-patch14-ensemble"
BUCKET_FILES = ("fire", "smoke", "flammable", "safe")  # RemoteFireHazardDetector order
SERVER_THRESHOLD = 0.1  # OWLv2ONNXDetector.DEFAULT_THRESHOLD
HAL_THRESHOLD = 0.3  # FIRE_HAZARD_CONFIDENCE_THRESHOLD

TOL_RAW = 1e-4
TOL_LABEL_PROB = 0.05  # labels must agree where the top prob is at least this
TOL_IOU = 0.999
TOL_CONF = 1e-3


def load_prompts(prompts_dir: Path) -> tuple[list[str], dict[str, str]]:
    classes, bucket_of = [], {}
    for bucket in BUCKET_FILES:
        for line in (prompts_dir / f"{bucket}_classes.txt").read_text().splitlines():
            if line.strip():
                classes.append(line.strip())
                bucket_of[line.strip()] = bucket
    return classes, bucket_of


def postprocess(raw: list[np.ndarray], classes: list[str], hw: tuple[int, int]):
    """The server's own post-processing (ONNXObjectDetector._postprocess_batch) on a stub detector."""
    stub = SimpleNamespace(
        _threshold=SERVER_THRESHOLD, _nms=True,
        _get_label_pool=lambda c: c, _to_orig_normalized=unowlv2_boxes,
    )
    return ONNXObjectDetector._postprocess_batch(stub, raw, classes, [hw])[0]


def iou_xywh(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx1, by1, bx2, by2 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    inter = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    union = a[2] * a[3] + b[2] * b[3] - inter
    return float(inter / union) if union > 0 else 0.0


def compare_detections(old, new) -> tuple[list[str], dict]:
    errors = []
    if Counter(old.class_names) != Counter(new.class_names):
        errors.append(f"class multiset differs: old={Counter(old.class_names)} new={Counter(new.class_names)}")
        return errors, {}
    used, min_iou, max_dconf = set(), 1.0, 0.0
    for i, name in enumerate(old.class_names):
        cands = [j for j, n in enumerate(new.class_names) if n == name and j not in used]
        j = max(cands, key=lambda k: iou_xywh(old.bbox_xywh[i], new.bbox_xywh[k]))
        used.add(j)
        min_iou = min(min_iou, iou_xywh(old.bbox_xywh[i], new.bbox_xywh[j]))
        max_dconf = max(max_dconf, abs(float(old.confidence[i]) - float(new.confidence[j])))
    if min_iou < TOL_IOU:
        errors.append(f"matched box IoU {min_iou:.5f} < {TOL_IOU}")
    if max_dconf > TOL_CONF:
        errors.append(f"matched |dconf| {max_dconf:.2e} > {TOL_CONF}")
    return errors, {"min_iou": min_iou, "max_dconf": max_dconf}


def hal_buckets(det, bucket_of: dict[str, str]) -> Counter:
    return Counter(bucket_of[n] for n, c in zip(det.class_names, det.confidence) if c >= HAL_THRESHOLD)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", required=True, help="old single graph (prod owlv2_raw.onnx)")
    ap.add_argument("--text", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--images", default=str(ROOT / "tests/fixtures/images/fire-*.jpg"))
    ap.add_argument("--prompts-dir", default=str(ROOT.parent.parent / "hal/drivers/sensing/perceptions/processors/resources"))
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    classes, bucket_of = load_prompts(Path(args.prompts_dir))
    tokens = Owlv2Processor.from_pretrained(HF_MODEL_ID)(
        text=[f"a photo of {c}" for c in classes], images=None, return_tensors="np"
    )["input_ids"].astype(np.int64)
    print(f"{len(classes)} prompts, tokens {tokens.shape}")

    providers = ["CPUExecutionProvider"]
    old = ort.InferenceSession(args.old, providers=providers)
    text = ort.InferenceSession(args.text, providers=providers)
    image = ort.InferenceSession(args.image, providers=providers)

    t0 = time.perf_counter()
    query_embeds = text.run(None, {"class_tokens": tokens})[0]
    text_ms = (time.perf_counter() - t0) * 1e3

    paths = sorted(glob.glob(args.images))
    if not paths:
        print(f"no images match {args.images}")
        return 1
    report, failed = {"prompts": len(classes), "text_graph_ms": round(text_ms, 1), "images": {}}, False
    for path in paths:
        img = cv2.imread(path)
        hw = img.shape[:2]
        x = owlv2_preprocess(img)[0][None]

        t0 = time.perf_counter()
        raw_old = old.run(None, {"images": x, "class_tokens": tokens})
        old_ms = (time.perf_counter() - t0) * 1e3
        t0 = time.perf_counter()
        raw_new = image.run(None, {"images": x, "query_embeds": query_embeds})
        new_ms = (time.perf_counter() - t0) * 1e3

        errors = []
        d_boxes = float(np.abs(raw_old[0] - raw_new[0]).max())
        d_probs = float(np.abs(raw_old[1] - raw_new[1]).max())
        if d_boxes > TOL_RAW:
            errors.append(f"raw boxes max abs {d_boxes:.2e} > {TOL_RAW}")
        if d_probs > TOL_RAW:
            errors.append(f"raw probs max abs {d_probs:.2e} > {TOL_RAW}")
        confident = raw_old[1].max(-1) >= TOL_LABEL_PROB
        label_mismatch = int(((raw_old[2] != raw_new[2]) & confident).sum())
        if label_mismatch:
            errors.append(f"{label_mismatch} labels differ where max prob >= {TOL_LABEL_PROB}")

        det_old, det_new = postprocess(raw_old, classes, hw), postprocess(raw_new, classes, hw)
        det_errors, det_stats = compare_detections(det_old, det_new)
        errors += det_errors
        b_old, b_new = hal_buckets(det_old, bucket_of), hal_buckets(det_new, bucket_of)
        if b_old != b_new:
            errors.append(f"HAL buckets differ: old={dict(b_old)} new={dict(b_new)}")

        failed |= bool(errors)
        name = Path(path).name
        print(f"{'FAIL' if errors else 'PASS'} {name}: raw boxes {d_boxes:.2e} probs {d_probs:.2e} | "
              f"{len(det_old.class_names)} dets | HAL {dict(b_old)} | old {old_ms:.0f} ms, image graph {new_ms:.0f} ms")
        for e in errors:
            print(f"    {e}")
        report["images"][name] = {
            "pass": not errors, "errors": errors, "raw_boxes_max_abs": d_boxes, "raw_probs_max_abs": d_probs,
            "detections": len(det_old.class_names), "hal_buckets": dict(b_old), **det_stats,
            "old_ms": round(old_ms, 1), "image_graph_ms": round(new_ms, 1),
        }

    print(f"text graph (all {len(classes)} prompts, once): {text_ms:.0f} ms")
    print("RESULT:", "FAIL" if failed else "PASS")
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
