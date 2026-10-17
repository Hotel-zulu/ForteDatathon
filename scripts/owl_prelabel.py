"""Черновая разметка «по описанию» моделью OWLv2 (Apache-2.0, работает локально, веса в models/).

Находит то, чего нет в COCO: деньги, удостоверение, сумку инкассатора, открытый сейф и т.п.

    python scripts/owl_prelabel.py --test X-1_021.000.jpg X-17_007.000.jpg   # проверка на кадрах -> workdir/owl_test/
    python scripts/owl_prelabel.py --push            # добавить черновики во ВСЕ неразмеченные задачи «Объектов»

Черновики добавляются как predictions (их видно серым/пунктиром), разметку людей не трогают.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
MODEL_DIR = ROOT / "models" / "owlv2-base-patch16-ensemble"
IMG_DIR = ROOT / "annotation" / "objects" / "images"

# класс -> (текстовые запросы, порог уверенности)
QUERIES = {
    "cash": (["banknotes", "paper money", "stack of cash", "money bills in hand"], 0.20),
    "id_card": (["identity card", "passport", "ID document"], 0.28),
    "phone": (["smartphone", "mobile phone in hand"], 0.25),
    "collector_bag": (["green cash bag", "sealed money bag"], 0.22),
    "personal_item": (["handbag", "jacket", "backpack", "wallet"], 0.25),
    "food_drink": (["cup", "coffee cup", "bottle", "food"], 0.25),
    "safe_open": (["open safe door", "open metal cabinet"], 0.25),
}


def device():
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


class Owl:
    def __init__(self):
        from transformers import Owlv2ForObjectDetection, Owlv2Processor
        self.proc = Owlv2Processor.from_pretrained(MODEL_DIR)
        self.dev = device()
        self.model = Owlv2ForObjectDetection.from_pretrained(MODEL_DIR).to(self.dev).eval()
        self.texts, self.owner, self.thr = [], [], []
        for cls, (qs, thr) in QUERIES.items():
            for q in qs:
                self.texts.append(q); self.owner.append(cls); self.thr.append(thr)

    @torch.no_grad()
    def __call__(self, img: Image.Image):
        W, H = img.size
        inp = self.proc(text=[self.texts], images=img, return_tensors="pt").to(self.dev)
        out = self.model(**inp)
        # OWLv2 паддит картинку до квадрата (сверху-слева): координаты относительно max(W,H).
        # Постобработка вручную — не зависит от версии transformers.
        S = max(W, H)
        probs = out.logits[0].sigmoid().float().cpu()            # [N боксов, Q запросов]
        boxes = out.pred_boxes[0].float().cpu()                   # cx, cy, w, h в долях квадрата
        scores, labels = probs.max(dim=-1)
        dets = []
        for (cx, cy, bw, bh), score, lab in zip(boxes.tolist(), scores.tolist(), labels.tolist()):
            cls = self.owner[lab]
            if score < self.thr[lab]:
                continue
            x1, y1, x2, y2 = [max(0, v * S) for v in (cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2)]
            x2, y2 = min(x2, W), min(y2, H)
            if x2 - x1 < 3 or y2 - y1 < 3 or (x2 - x1) * (y2 - y1) > 0.5 * W * H:
                continue
            dets.append({"cls": cls, "score": score, "box": (x1, y1, x2, y2), "query": self.texts[lab]})
        return nms(dets)


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def nms(dets, thr=0.4):
    dets = sorted(dets, key=lambda d: -d["score"])
    keep = []
    for d in dets:
        if all(iou(d["box"], k["box"]) < thr for k in keep):
            keep.append(d)
    return keep


COLORS = {"cash": "#43a047", "id_card": "#fdd835", "phone": "#e53935", "collector_bag": "#9ccc65",
          "personal_item": "#8d6e63", "food_drink": "#ff7043", "safe_open": "#d81b60"}


def test(names):
    owl = Owl()
    out = ROOT / "workdir" / "owl_test"
    out.mkdir(parents=True, exist_ok=True)
    for n in names:
        img = Image.open(IMG_DIR / n).convert("RGB")
        dets = refine(owl(img), img, n.split("_")[0], load_glass())
        big = img.resize((img.width * 2, img.height * 2))
        d = ImageDraw.Draw(big)
        for x in dets:
            b = [v * 2 for v in x["box"]]
            d.rectangle(b, outline=COLORS[x["cls"]], width=3)
            d.text((b[0] + 2, b[1] - 12), f'{x["cls"]} {x["score"]:.2f}', fill=COLORS[x["cls"]])
        big.save(out / n)
        print(n, [(x["cls"], round(x["score"], 2)) for x in dets])


def load_glass(path=ROOT / "annotation" / "zones" / "export.json"):
    """{video: [полигоны стекла в процентах]} из экспорта зон (если есть)."""
    if not path.exists():
        print("⚠ Нет экспорта зон — фильтр отражений в стекле не применяется")
        return {}
    out = {}
    for t in json.loads(path.read_text(encoding="utf-8")):
        anns = t.get("annotations") or []
        if not anns:
            continue
        polys = [r["value"]["points"] for r in anns[-1]["result"]
                 if r.get("type") == "polygonlabels" and r["value"]["polygonlabels"][0] == "glass_wall"]
        if polys:
            out[t["data"]["video"]] = polys
    print(f"Полигоны стекла: {len(out)} видео")
    return out


def inside(pt, poly):
    x, y = pt
    c = False
    for i in range(len(poly)):
        x1, y1 = poly[i]; x2, y2 = poly[i - 1]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-9) + x1:
            c = not c
    return c


def is_green(img, box):
    """Сумка инкассатора зелёная: доля зелёных пикселей в рамке."""
    crop = img.crop(tuple(int(v) for v in box)).convert("HSV").resize((24, 24))
    px = list(crop.getdata())
    # сумка в датасете бледно-оливковая: тон ~45–50 (шкала PIL 0–255), насыщенность ~50
    green = sum(1 for h, s, v in px if 36 <= h <= 115 and s > 30 and v > 50)
    return green / len(px) > 0.30


def refine(dets, img, video, glass):
    W, H = img.size
    out = []
    for x in dets:
        x1, y1, x2, y2 = x["box"]
        cx, cy = 100 * (x1 + x2) / 2 / W, 100 * (y1 + y2) / 2 / H
        if any(inside((cx, cy), p) for p in glass.get(video, [])):
            continue  # в зоне стекла: отражение или предмет за стеклом
        if x["cls"] in ("personal_item", "collector_bag"):
            # сумка инкассатора = зелёная и найдена уверенно; иначе — личная вещь
            x["cls"] = "collector_bag" if (is_green(img, x["box"]) and x["score"] >= 0.35) else "personal_item"
        out.append(x)
    return out


def push():
    import label_studio as L  # scripts/label_studio.py
    owl = Owl()
    glass = load_glass()
    proj = L.find_project(L.TITLES["objects"])
    page, added = 1, 0
    while True:
        r = L.api("GET", "/api/tasks", params={"project": proj["id"], "page": page, "page_size": 100})
        tasks = r.get("tasks", r) if isinstance(r, dict) else r
        if not tasks:
            break
        for t in tasks:
            if t.get("total_annotations") or t.get("annotations"):
                continue  # уже размечено человеком — не трогаем
            path = ROOT / t["data"]["image"].split("d=")[-1]
            img = Image.open(path).convert("RGB")
            W, H = img.size
            result = []
            for i, x in enumerate(refine(owl(img), img, t["data"].get("video"), glass)):
                x1, y1, x2, y2 = x["box"]
                result.append({"id": f"owl{t['id']}_{i}", "from_name": "box", "to_name": "image",
                               "type": "rectanglelabels", "original_width": W, "original_height": H,
                               "score": x["score"],
                               "value": {"x": 100 * x1 / W, "y": 100 * y1 / H, "width": 100 * (x2 - x1) / W,
                                         "height": 100 * (y2 - y1) / H, "rotation": 0,
                                         "rectanglelabels": [x["cls"]]}})
            if result:
                L.api("POST", "/api/predictions", json={"task": t["id"], "model_version": "owlv2",
                                                        "result": result})
                added += 1
        page += 1
        print(f"обработано страниц: {page - 1}, задач с черновиками: {added}")
        if len(tasks) < 100:
            break
    print(f"Готово: черновики OWLv2 добавлены в {added} задач")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", nargs="*")
    ap.add_argument("--push", action="store_true")
    a = ap.parse_args()
    if a.test:
        test(a.test)
    elif a.push:
        sys.path.insert(0, str(ROOT / "scripts"))
        push()
    else:
        print(__doc__)
