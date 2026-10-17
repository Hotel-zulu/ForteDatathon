"""Обучение детектора RT-DETRv2 (Apache-2.0) на нашей разметке — единственный детектор сервиса.

    python scripts/build_dataset.py --val X-3 X-10 X-33 X-39     # txt-разметка: cls cx cy w h
    python scripts/train_rtdetr.py --epochs 40                    # ~3–4 ч на M4 (MPS)
    python scripts/train_rtdetr.py --epochs 1 --limit 32          # проверка, что всё запускается

Веса: models/rtdetr_cashier/ (лучшая эпоха по mAP50 на val). Базовая модель: PekingU/rtdetr_v2_r18vd
(Apache-2.0, COCO) в models/rtdetr_v2_r18vd. Ни ultralytics, ни других AGPL-компонентов здесь нет.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from torchvision import tv_tensors  # noqa: E402
from torchvision.transforms import v2 as T  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DS = ROOT / "datasets" / "cashier"
BASE = ROOT / "models" / "rtdetr_v2_r18vd"
OUT = ROOT / "models" / "rtdetr_cashier"


def read_names():
    names = {}
    for line in (DS / "data.yaml").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line[:1].isdigit() and ":" in line:
            k, v = line.split(":", 1)
            names[int(k)] = v.strip()
    return [names[i] for i in sorted(names)]


class LabelDS(torch.utils.data.Dataset):
    def __init__(self, split, train, limit=None, size=800):
        splits = split if isinstance(split, (list, tuple)) else [split]
        self.imgs = sorted(p for sp in splits for p in (DS / "images" / sp).glob("*.jpg"))[:limit]
        self.train, self.size = train, size
        self.aug = T.Compose([
            T.RandomPhotometricDistort(p=0.8),
            T.RandomZoomOut(fill={tv_tensors.Image: (114, 114, 114), "others": 0}, side_range=(1.0, 1.6), p=0.3),
            T.RandomIoUCrop(min_scale=0.5, sampler_options=[0.3, 0.5, 0.7, 0.9, 1.0]),
            T.RandomHorizontalFlip(),
            T.RandomAffine(degrees=12, translate=(0.08, 0.08), scale=(0.7, 1.3), shear=6),   # другой ракурс
            T.RandomPerspective(distortion_scale=0.3, p=0.4),                                 # камера сбоку
            T.SanitizeBoundingBoxes(min_size=2),
        ])

    def __len__(self):
        return len(self.imgs)

    def load(self, i):
        p = self.imgs[i]
        img = Image.open(p).convert("RGB")
        W, H = img.size
        lab = (DS / "labels" / p.parent.name / p.with_suffix(".txt").name)   # images/<split>/x.jpg -> labels/<split>/x.txt
        boxes, cls = [], []
        for line in (lab.read_text().split("\n") if lab.exists() else []):
            if not line.strip():
                continue
            c, cx, cy, w, h = map(float, line.split())
            boxes.append([(cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H])
            cls.append(int(c))
        return img, torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4), torch.tensor(cls, dtype=torch.int64)

    def __getitem__(self, i):
        img, boxes, cls = self.load(i)
        im = tv_tensors.Image(T.functional.pil_to_tensor(img))
        bb = tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=(img.height, img.width))
        if self.train:
            for _ in range(5):   # RandomIoUCrop может выкинуть все рамки — пробуем ещё раз
                o = self.aug({"image": im, "boxes": bb, "labels": cls})
                if len(o["boxes"]) or not len(cls):
                    break
            im, bb, cls = o["image"], o["boxes"], o["labels"]
        H, W = im.shape[-2:]
        im = T.functional.resize(im, [self.size, self.size], antialias=True).float() / 255.0
        b = bb.as_subclass(torch.Tensor).clone()
        # -> нормированные cx, cy, w, h (формат RT-DETR)
        tgt = torch.stack([(b[:, 0] + b[:, 2]) / 2 / W, (b[:, 1] + b[:, 3]) / 2 / H,
                           (b[:, 2] - b[:, 0]) / W, (b[:, 3] - b[:, 1]) / H], 1) if len(b) else torch.zeros(0, 4)
        return im, {"class_labels": cls, "boxes": tgt.clamp(0, 1)}


def collate(b):
    return torch.stack([x[0] for x in b]), [x[1] for x in b]


def ap50(preds, gts, ncls):
    """mAP@0.5 по классам. preds: [(img, cls, score, box)], gts: {(img, cls): [boxes]} — xyxy нормированные."""
    def iou(a, b):
        ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
        u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
        return ix * iy / (u + 1e-9)
    out = {}
    for c in range(ncls):
        npos = sum(len(v) for (i, k), v in gts.items() if k == c)
        if not npos:
            continue
        used, tp = set(), []
        for img, k, s, box in sorted([p for p in preds if p[1] == c], key=lambda p: -p[2]):
            best, bj = 0, None
            for j, g in enumerate(gts.get((img, c), [])):
                v = iou(box, g)
                if v > best and (img, c, j) not in used:
                    best, bj = v, j
            if best >= 0.5:
                used.add((img, c, bj)); tp.append(1)
            else:
                tp.append(0)
        tp = np.array(tp); fp = 1 - tp
        rec = np.cumsum(tp) / npos if len(tp) else np.array([0])
        prec = np.cumsum(tp) / np.maximum(np.cumsum(tp) + np.cumsum(fp), 1) if len(tp) else np.array([0])
        ap = 0
        for t in np.linspace(0, 1, 101):
            ap += (prec[rec >= t].max() if (rec >= t).any() else 0) / 101
        out[c] = ap
    return out


@torch.no_grad()
def evaluate(model, dl, dev, names):
    model.eval()
    preds, gts, n = [], {}, 0
    for x, ys in dl:
        o = model(pixel_values=x.to(dev))
        prob = o.logits.sigmoid().cpu(); bx = o.pred_boxes.cpu()
        for b in range(len(ys)):
            img = n + b
            for k, g in zip(ys[b]["class_labels"].tolist(), ys[b]["boxes"].tolist()):
                cx, cy, w, h = g
                gts.setdefault((img, k), []).append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
            s, idx = prob[b].flatten().topk(100)
            for sc, ii in zip(s.tolist(), idx.tolist()):
                q, k = divmod(ii, prob.shape[-1])
                if sc < 0.05:
                    break
                cx, cy, w, h = bx[b, q].tolist()
                preds.append((img, k, sc, [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]))
        n += len(ys)
    per = ap50(preds, gts, len(names))
    model.train()
    return (float(np.mean(list(per.values()))) if per else 0.0), {names[k]: round(v, 3) for k, v in per.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--size", type=int, default=800)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--init", help="продолжить с сохранённой модели (например models/rtdetr_cashier)")
    ap.add_argument("--all", action="store_true", help="финальная модель: val тоже в обучение, сохраняется последняя эпоха")
    ap.add_argument("--rare-x", type=float, default=3.0, help="во сколько раз чаще показывать кадры с редкими классами")
    a = ap.parse_args()
    from transformers import RTDetrV2ForObjectDetection

    names = read_names()
    dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0); random.seed(0)
    if a.init:   # продолжение обучения с нашей модели (те же классы)
        import shutil
        shutil.copytree(a.init, ROOT / "models" / "rtdetr_cashier_resume_from", dirs_exist_ok=True)
        model = RTDetrV2ForObjectDetection.from_pretrained(ROOT / "models" / "rtdetr_cashier_resume_from").to(dev)
    else:
        model = RTDetrV2ForObjectDetection.from_pretrained(
            BASE, num_labels=len(names), id2label=dict(enumerate(names)), label2id={n: i for i, n in enumerate(names)},
            ignore_mismatched_sizes=True).to(dev)
    tr = LabelDS(["train", "val"] if a.all else "train", True, a.limit, a.size)
    va = LabelDS("val", False, a.limit, a.size)
    # редкие классы (документы, личные вещи, деньги в счётчике, сумка, открытый сейф, удостоверение) — чаще
    rare = {names.index(n) for n in ("documents", "personal_item", "cash_in_counter", "collector_bag", "safe_open",
                                     "id_card", "door_open", "key_in_lock") if n in names}
    def _w(p):
        lab = DS / "labels" / p.parent.name / p.with_suffix(".txt").name
        cls = {int(l.split()[0]) for l in lab.read_text().split("\n") if l.strip()} if lab.exists() else set()
        return a.rare_x if cls & rare else 1.0
    weights = [_w(p) for p in tr.imgs]
    sampler = torch.utils.data.WeightedRandomSampler(weights, num_samples=len(tr), replacement=True)
    dl = torch.utils.data.DataLoader(tr, batch_size=a.batch, sampler=sampler, collate_fn=collate, num_workers=2,
                                     persistent_workers=True)
    print(f"кадров с редкими классами: {sum(w > 1 for w in weights)} из {len(weights)} (×{a.rare_x})", flush=True)
    # persistent_workers: процессы загрузки создаются один раз (не перечитывают скрипт после каждой эпохи)
    dv = torch.utils.data.DataLoader(va, batch_size=a.batch, collate_fn=collate, num_workers=2, persistent_workers=True)
    bb = [p for n, p in model.named_parameters() if "backbone" in n]
    rest = [p for n, p in model.named_parameters() if "backbone" not in n]
    opt = torch.optim.AdamW([{"params": bb, "lr": a.lr / 10}, {"params": rest, "lr": a.lr}], weight_decay=1e-4)
    steps = a.epochs * len(dl); warm = min(300, steps // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else
                                              0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))) * 0.95 + 0.05)
    OUT.mkdir(parents=True, exist_ok=True)
    best, log = -1, []
    print(f"train {len(tr)}  val {len(va)}  классов {len(names)}  устройство {dev}", flush=True)
    for ep in range(1, a.epochs + 1):
        t0, tot = time.time(), 0.0
        for i, (x, ys) in enumerate(dl):
            ys = [{k: v.to(dev) for k, v in y.items()} for y in ys]
            loss = model(pixel_values=x.to(dev), labels=ys).loss
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            opt.step(); sched.step()
            tot += loss.item()
            if i % 20 == 0:
                print(f"  ep {ep} it {i}/{len(dl)} loss {loss.item():.3f}", flush=True)
        m, per = evaluate(model, dv, dev, names)
        log.append({"epoch": ep, "loss": round(tot / len(dl), 4), "map50": round(m, 4), "per_class": per,
                    "sec": round(time.time() - t0)})
        print(f"эпоха {ep}: loss {tot / len(dl):.3f}  val mAP50 {m:.3f}  {per}  ({time.time() - t0:.0f} c)", flush=True)
        if m > best or a.all:   # --all: валидация видела эти кадры, берём последнюю эпоху
            best = m
            model.save_pretrained(OUT)
            print(f"  ✓ сохранено (лучшее {best:.3f})", flush=True)
        (OUT / "train_log.json").write_text(json.dumps(log, ensure_ascii=False, indent=1))
    print(f"Готово. Лучшее val mAP50 {best:.3f}. Веса: {OUT}")


if __name__ == "__main__":
    main()
