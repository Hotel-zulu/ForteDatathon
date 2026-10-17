"""Интерфейс детекторов и верификатора.

Архитектура (см. docs/ARCHITECTURE.md):
  кадры -> Perception (люди/руки/предметы/трекинг, по каждому кадру)
        -> Rule-детекторы (по таймлайну) -> Candidate
        -> Verifier (локальная VLM: подтвердить/отклонить + описание)
        -> Violation (+ норматив из regulations.yaml + кадры-доказательства)

Чтобы добавить нарушение: создать класс-наследник Detector в src/detectors/, указать type_id
(ключ из regulations.yaml) и зарегистрировать его в src/detectors/__init__.py.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass
class FrameObs:
    """Наблюдения на одном кадре (заполняет perception-модуль)."""
    t: float
    persons: list[dict] = field(default_factory=list)   # {"track_id", "xyxy"(0..1), "role": cashier|client|collector|unknown, "keypoints"}
    objects: list[dict] = field(default_factory=list)   # {"cls": phone|cash|bag|drawer_open|..., "xyxy", "conf"}
    extra: dict = field(default_factory=dict)


@dataclass
class Candidate:
    type_id: str                 # ключ из configs/regulations.yaml
    t_start: float
    t_end: float
    t_peak: float                # таймкод для ответа (самый показательный кадр)
    confidence: float
    observed: str                # что видно, фактами, без домыслов
    boxes: list[dict] = field(default_factory=list)
    evidence_ts: list[float] = field(default_factory=list)


class Detector(ABC):
    type_id: str = ""

    def reset(self) -> None:
        """Вызывается перед каждым видео."""

    def update(self, obs: FrameObs, frame: np.ndarray) -> None:
        """Потоковая обработка кадра (по умолчанию ничего)."""

    @abstractmethod
    def finalize(self, timeline: list[FrameObs]) -> list[Candidate]:
        """Вернуть кандидатов по всему таймлайну."""


class Verifier(ABC):
    """Второй шаг (например, локальная VLM). Снижает ложные срабатывания."""

    @abstractmethod
    def verify(self, video_path: str, cand: Candidate) -> tuple[bool, float, str | None]:
        """-> (подтверждено?, уверенность, уточнённое описание или None)"""


class Perception(ABC):
    @abstractmethod
    def __call__(self, t: float, frame: np.ndarray) -> FrameObs: ...
