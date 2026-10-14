"""OWLv2 text/image split must match the old single graph (tiny random model, no download)."""

import numpy as np
import onnxruntime as ort
import pytest
import torch
from transformers import Owlv2Config, Owlv2ForObjectDetection

from core.export.entries.export_owlv2 import (
    OWLv2ImageONNX,
    OWLv2TextONNX,
    _export_image,
    _export_text,
)
from core.export.utils.nms import xyxy_to_xywh_normalized

BOS, EOS = 49406, 49407


class _OldOWLv2ONNX(torch.nn.Module):
    """Frozen copy of the pre-split single-graph wrapper (nms=False): the reference."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, images, class_tokens):
        attention_mask = (class_tokens != 0).long()
        outputs = self.model(input_ids=class_tokens, attention_mask=attention_mask, pixel_values=images)
        probs = outputs.logits.sigmoid()
        cx, cy, w, h = outputs.pred_boxes.unbind(-1)
        boxes_xyxy = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)
        xywh = xyxy_to_xywh_normalized(boxes_xyxy, input_hw=(1, 1))
        _, labels = probs.max(dim=-1)
        return xywh, probs, labels


@pytest.fixture(scope="module")
def tiny():
    torch.manual_seed(0)
    cfg = Owlv2Config(
        text_config={"hidden_size": 32, "num_hidden_layers": 1, "num_attention_heads": 2,
                     "intermediate_size": 37, "max_position_embeddings": 16, "vocab_size": 49408},
        vision_config={"hidden_size": 32, "num_hidden_layers": 1, "num_attention_heads": 2,
                       "intermediate_size": 37, "image_size": 56, "patch_size": 14},
        projection_dim=32,
    )
    model = Owlv2ForObjectDetection(cfg).eval()
    # 3 queries of different lengths, padded with 0 like the OWLv2 tokenizer does.
    tokens = torch.zeros(3, 16, dtype=torch.long)
    for i, n in enumerate((4, 7, 10)):
        tokens[i, 0] = BOS
        tokens[i, 1:n - 1] = torch.randint(1, 49000, (n - 2,))
        tokens[i, n - 1] = EOS
    image = torch.rand(1, 3, 56, 56)
    return model, tokens, image


def test_text_wrapper_matches_hf_text_embeds(tiny):
    model, tokens, image = tiny
    with torch.no_grad():
        ours = OWLv2TextONNX(model)(tokens)
        hf = model.owlv2(input_ids=tokens, pixel_values=image, attention_mask=(tokens != 0).long()).text_embeds
    torch.testing.assert_close(ours, hf, atol=1e-6, rtol=0)


def test_image_wrapper_matches_old_single_graph(tiny):
    model, tokens, image = tiny
    with torch.no_grad():
        old = _OldOWLv2ONNX(model)(image, tokens)
        new = OWLv2ImageONNX(model)(image, OWLv2TextONNX(model)(tokens))
    torch.testing.assert_close(new[0], old[0], atol=1e-5, rtol=0)
    torch.testing.assert_close(new[1], old[1], atol=1e-5, rtol=0)
    assert torch.equal(new[2], old[2])


def test_image_wrapper_broadcasts_queries_over_batch(tiny):
    model, tokens, image = tiny
    with torch.no_grad():
        q = OWLv2TextONNX(model)(tokens)
        one = OWLv2ImageONNX(model)(image, q)
        two = OWLv2ImageONNX(model)(image.repeat(2, 1, 1, 1), q)
    for a, b in zip(one, two):
        assert b.shape[0] == 2
        torch.testing.assert_close(b[0], a[0], atol=1e-6, rtol=0)
        torch.testing.assert_close(b[1], a[0], atol=1e-6, rtol=0)


def test_onnx_export_roundtrip(tiny, tmp_path):
    model, tokens, image = tiny
    text_path, image_path = tmp_path / "text.onnx", tmp_path / "image.onnx"
    _export_text(model, tokens, text_path, opset=17)
    with torch.no_grad():
        ref_q = OWLv2TextONNX(model)(tokens)
        ref = OWLv2ImageONNX(model)(image, ref_q)
    _export_image(model, image, ref_q, image_path, opset=17)

    providers = ["CPUExecutionProvider"]
    text = ort.InferenceSession(str(text_path), providers=providers)
    img = ort.InferenceSession(str(image_path), providers=providers)
    assert [i.name for i in text.get_inputs()] == ["class_tokens"]
    assert [i.name for i in img.get_inputs()] == ["images", "query_embeds"]

    # A different query count than the export dummy proves the query axis is dynamic.
    q = text.run(None, {"class_tokens": tokens[:2].numpy()})[0]
    np.testing.assert_allclose(q, ref_q[:2].numpy(), atol=1e-4)
    boxes, probs, _labels = img.run(None, {"images": image.numpy(), "query_embeds": q})
    with torch.no_grad():
        ref2 = OWLv2ImageONNX(model)(image, ref_q[:2])
    np.testing.assert_allclose(boxes, ref2[0].numpy(), atol=1e-4)
    np.testing.assert_allclose(probs, ref2[1].numpy(), atol=1e-4)
    assert probs.shape[-1] == 2
    assert ref[1].shape[-1] == 3
