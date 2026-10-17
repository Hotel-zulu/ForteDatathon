"""Формат ответа — строго по разделу 10 ТЗ (оценка автоматическая, лишнее/битое поле = потеря баллов)."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class EvidenceFrame(BaseModel):
    timestamp_sec: float
    image_base64: str  # JPEG в base64, без префикса data:


class Violation(BaseModel):
    type: str
    timecode_sec: float
    description: str
    evidence_frames: list[EvidenceFrame] = Field(default_factory=list, max_length=3)
    # Дополнительные поля (схема жюри их не требует, но они нужны UI и для критерия 4).
    # Если окажется, что лишние поля мешают автопроверке — уберите их в api/main.py (strip_extra).
    rule: str | None = None
    source: str | None = None
    confidence: float | None = None


class JobResult(BaseModel):
    job_id: str
    status: Literal["processing", "done", "failed"]
    verdict: Literal["violation", "no_violation"] | None = None
    violations: list[Violation] = Field(default_factory=list)
    error: str | None = None
