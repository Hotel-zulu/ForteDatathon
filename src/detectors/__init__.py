"""Реестр детекторов. Добавляйте сюда новые классы после реализации."""
from src.detectors.base import Candidate, Detector, FrameObs, Perception, Verifier
from src.detectors.examples import LeavingWorkplaceDetector, NullPerception, PhoneDetector


def build_perception() -> Perception:
    # TODO: заменить на реальный модуль: детектор людей/предметов (Apache-2.0: RT-DETR / D-FINE / YOLOX)
    #       + трекинг (ByteTrack) + поза (RTMPose). Веса — в models/, без интернета.
    return NullPerception()


def build_detectors() -> list[Detector]:
    return [LeavingWorkplaceDetector(), PhoneDetector()]


def build_verifier() -> Verifier | None:
    # TODO: локальная VLM (проверьте лицензию на коммерческое использование). None = без верификации.
    return None


__all__ = ["Candidate", "Detector", "FrameObs", "Perception", "Verifier",
           "build_perception", "build_detectors", "build_verifier"]
