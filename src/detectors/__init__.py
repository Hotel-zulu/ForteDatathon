"""Реестр детекторов. Добавляйте сюда новые классы после реализации."""
from src.detectors.base import Candidate, Detector, FrameObs, Perception, Verifier
from src.detectors.examples import NullPerception


def build_perception() -> Perception:
    """PERCEPTION=rtdetr (по умолчанию: RT-DETRv2, Apache-2.0) | null (заглушка для проверки сервиса)."""
    import os
    if os.getenv("PERCEPTION", "rtdetr") == "rtdetr":
        try:
            from src.detectors.rtdetr_perception import RTDetrPerception
            return RTDetrPerception()
        except (ImportError, OSError) as e:
            print(f"[perception] RT-DETR недоступен ({e}) — использую заглушку")
    return NullPerception()


def build_detectors() -> list[Detector]:
    """Все правила. Срабатывают только типы с enabled: true в configs/regulations.yaml."""
    from src.detectors import rules as R
    return [R.DepartureDetector(), R.CashAtNextClient(), R.SafeLeftOpen(), R.DoorLeftOpen(), R.NoIdentification(),
            R.PersonalItems(), R.ThirdPerson(), R.NotCountedToSafe(), R.NoMachineCheck(), R.ClientLeftEarly()]


def build_verifier() -> Verifier | None:
    # TODO: локальная VLM (проверьте лицензию на коммерческое использование). None = без верификации.
    return None


__all__ = ["Candidate", "Detector", "FrameObs", "Perception", "Verifier",
           "build_perception", "build_detectors", "build_verifier"]
