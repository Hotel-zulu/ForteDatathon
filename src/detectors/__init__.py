"""Реестр детекторов. Добавляйте сюда новые классы после реализации."""
from src.detectors.base import Candidate, Detector, FrameObs, Perception, Verifier
from src.detectors.examples import NullPerception


def build_perception() -> Perception:
    """PERCEPTION=cashier (по умолчанию: наша модель + люди) | yolo (только COCO) | null.

    TODO к финалу: Apache-2.0 детектор (RT-DETR / D-FINE). Веса — в models/, офлайн.
    """
    import os
    kind = os.getenv("PERCEPTION", "cashier")
    try:
        if kind == "cashier":
            from src.detectors.cashier_perception import CashierPerception
            return CashierPerception()
        if kind == "rtdetr":   # Apache-2.0, без ultralytics — для финальной сдачи
            from src.detectors.rtdetr_perception import RTDetrPerception
            return RTDetrPerception()
        if kind == "yolo":
            from src.detectors.yolo_perception import YoloPerception
            return YoloPerception()
    except (ImportError, FileNotFoundError) as e:
        print(f"[perception] {kind} недоступен ({e}) — использую заглушку")
    return NullPerception()


def build_detectors() -> list[Detector]:
    from src.detectors.rules import (CashAtNextClient, DepartureDetector, DoorLeftOpen, NoIdentification,
                                     SafeLeftOpen)
    # NoIdentification срабатывает, только если тип no_client_identification включён в regulations.yaml
    return [DepartureDetector(), CashAtNextClient(), SafeLeftOpen(), DoorLeftOpen(), NoIdentification()]


def build_verifier() -> Verifier | None:
    # TODO: локальная VLM (проверьте лицензию на коммерческое использование). None = без верификации.
    return None


__all__ = ["Candidate", "Detector", "FrameObs", "Perception", "Verifier",
           "build_perception", "build_detectors", "build_verifier"]
