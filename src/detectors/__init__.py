"""Реестр детекторов. Добавляйте сюда новые классы после реализации."""
from src.detectors.base import Candidate, Detector, FrameObs, Perception, Verifier
from src.detectors.examples import LeavingWorkplaceDetector, NullPerception, PhoneDetector


def build_perception() -> Perception:
    """PERCEPTION=yolo (по умолчанию, если установлен ultralytics) | null.

    TODO к финалу: Apache-2.0 детектор (RT-DETR / D-FINE) + поза (RTMPose). Веса — в models/, офлайн.
    """
    import os
    kind = os.getenv("PERCEPTION", "yolo")
    if kind == "yolo":
        try:
            from src.detectors.yolo_perception import YoloPerception
            return YoloPerception()
        except ImportError:
            print("[perception] ultralytics не установлен — использую заглушку")
    return NullPerception()


def build_detectors() -> list[Detector]:
    return [LeavingWorkplaceDetector(), PhoneDetector()]


def build_verifier() -> Verifier | None:
    # TODO: локальная VLM (проверьте лицензию на коммерческое использование). None = без верификации.
    return None


__all__ = ["Candidate", "Detector", "FrameObs", "Perception", "Verifier",
           "build_perception", "build_detectors", "build_verifier"]
