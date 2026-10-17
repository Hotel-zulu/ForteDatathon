"""Perception без AGPL: RT-DETRv2 (Apache-2.0, transformers) + простой IoU-трекер людей.

  • объекты кассы — наша дообученная модель models/rtdetr_cashier (scripts/train_rtdetr.py);
  • люди — исходная COCO-модель models/rtdetr_v2_r18vd (класс person);
  • трекинг — жадное сопоставление по IoU (свой код, без ByteTrack/ultralytics).

Это единственный детектор сервиса (YOLO/ultralytics — AGPL — удалены из проекта).
"""
from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.config import MODELS_DIR  # noqa: E402
from src.detectors.base import FrameObs, Perception  # noqa: E402
from src.detectors.pixels import PixelFeatures  # noqa: E402


def _iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / (u + 1e-9)


class IoUTracker:
    """Жадное сопоставление рамок с треками предыдущих кадров; трек живёт max_age секунд без наблюдений."""

    def __init__(self, thr=0.3, max_age=1.0):
        self.thr, self.max_age = thr, max_age
        self.reset()

    def reset(self):
        self.tracks, self.next_id = {}, 1   # id -> (box, t_last)

    def __call__(self, t, boxes):
        self.tracks = {k: v for k, v in self.tracks.items() if t - v[1] <= self.max_age}
        pairs = sorted(((_iou(b, tb), i, k) for i, b in enumerate(boxes) for k, (tb, _) in self.tracks.items()), reverse=True)
        ids, used = [None] * len(boxes), set()
        for v, i, k in pairs:
            if v < self.thr:
                break
            if ids[i] is None and k not in used:
                ids[i] = k; used.add(k)
        for i, b in enumerate(boxes):
            if ids[i] is None:
                ids[i] = self.next_id; self.next_id += 1
            self.tracks[ids[i]] = (b, t)
        return ids


class _Det:
    def __init__(self, path, dev, size):
        from transformers import RTDetrV2ForObjectDetection
        self.model = RTDetrV2ForObjectDetection.from_pretrained(path).to(dev).eval()
        self.names = self.model.config.id2label
        self.dev, self.size = dev, size

    @torch.no_grad()
    def __call__(self, rgb: np.ndarray, conf: float, keep=None):
        # RT-DETR: квадрат size×size, только /255 без нормировки — как в scripts/train_rtdetr.py
        x = torch.from_numpy(rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        x = torch.nn.functional.interpolate(x, size=(self.size, self.size), mode="bilinear", antialias=True)
        o = self.model(pixel_values=x.to(self.dev))
        prob = o.logits.sigmoid()[0].cpu(); bx = o.pred_boxes[0].cpu()
        s, idx = prob.flatten().topk(100)
        out = []
        for sc, ii in zip(s.tolist(), idx.tolist()):
            if sc < conf:
                break
            q, k = divmod(ii, prob.shape[-1])
            name = self.names[k]
            if keep and name not in keep:
                continue
            cx, cy, w, h = bx[q].tolist()
            out.append((name, sc, [max(0, cx - w / 2), max(0, cy - h / 2), min(1, cx + w / 2), min(1, cy + h / 2)]))
        return _nms(out)


def _nms(dets, thr=0.6):
    keep = []
    for d in sorted(dets, key=lambda d: -d[1]):
        if all(d[0] != k[0] or _iou(d[2], k[2]) < thr for k in keep):
            keep.append(d)
    return keep


class RTDetrPerception(PixelFeatures, Perception):
    def __init__(self, conf: float = 0.3, size: int = 640):
        dev = os.getenv("DEVICE") or ("mps" if torch.backends.mps.is_available() else
                                      "cuda" if torch.cuda.is_available() else "cpu")
        self.obj = _Det(os.getenv("RTDETR_WEIGHTS", str(MODELS_DIR / "rtdetr_cashier")), dev, size)
        self.ppl = _Det(os.getenv("RTDETR_PERSON", str(MODELS_DIR / "rtdetr_v2_r18vd")), dev, size)
        self.tracker = IoUTracker()
        self.conf = conf
        self.reset()

    def __call__(self, t: float, frame: np.ndarray) -> FrameObs:
        rgb = frame[:, :, ::-1].copy()
        obs = FrameObs(t=t)
        for name, sc, b in self.obj(rgb, self.conf):
            obs.objects.append({"cls": name, "xyxy": b, "conf": round(sc, 3)})
        ppl = self.ppl(rgb, 0.35, keep={"person"})
        for (name, sc, b), tid in zip(ppl, self.tracker(t, [p[2] for p in ppl])):
            obs.persons.append({"track_id": tid, "xyxy": b, "conf": round(sc, 3), "role": "unknown"})
        self.add_pixels(frame, obs)
        return obs

    def reset(self):
        self.reset_pixels()
        if hasattr(self, "tracker"):
            self.tracker.reset()
