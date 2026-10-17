"""Обучение детектора на размеченных кадрах (Apple Silicon: MPS).

    python scripts/train_detector.py                      # YOLO11s, 80 эпох  (быстрый эксперимент)
    python scripts/train_detector.py --model rtdetr-l.pt  # RT-DETR

Аугментации заточены под главный риск — другой угол камеры на финальном видео:
повороты, перспектива, масштаб, сдвиг, яркость. Горизонтальный флип включён (сцена симметрична по смыслу).

⚠ Веса ultralytics — AGPL-3.0: для экспериментов. К финалу — Apache-2.0 модель (см. models/README.md).
Результат: models/detector/weights/best.pt
"""
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "models" / "yolo11s.pt"))
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--name", default="detector", help="папка в models/ (не перезаписывать рабочую модель)")
    a = ap.parse_args()

    from ultralytics import YOLO
    import torch
    device = "mps" if torch.backends.mps.is_available() else ("0" if torch.cuda.is_available() else "cpu")
    model = YOLO(a.model)
    model.train(
        data=str(ROOT / "datasets" / "cashier" / "data.yaml"),
        epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, device=device,
        project=str(ROOT / "models"), name=a.name, exist_ok=True,
        patience=25, workers=2, seed=0, plots=True,
        # аугментации под «другой ракурс»
        degrees=10, perspective=0.0008, scale=0.5, translate=0.15, shear=4,
        fliplr=0.5, hsv_v=0.4, hsv_s=0.5, mosaic=1.0, close_mosaic=10,
    )
    metrics = model.val()
    print("mAP50:", round(metrics.box.map50, 3), " mAP50-95:", round(metrics.box.map, 3))
    names = model.names
    for i, ap50 in enumerate(metrics.box.ap50):
        print(f"  {names[metrics.box.ap_class_index[i]]:18s} AP50={ap50:.3f}")
    print(f"Веса: {ROOT / 'models' / a.name / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
