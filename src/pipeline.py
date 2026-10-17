"""Главная функция: видео -> результат в формате ТЗ.

    python -m src.pipeline path/to/video.mov            # печатает JSON (без base64)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from src import config, rules
from src.detectors import Candidate, build_detectors, build_perception, build_verifier
from src.evidence import make_evidence
from src.video_io import iter_frames, probe


_PERCEPTION = None


def _perception():
    """Модель грузится один раз на процесс (а не на каждое видео)."""
    global _PERCEPTION
    if _PERCEPTION is None:
        _PERCEPTION = build_perception()
    return _PERCEPTION


def _dedup(cands: list[Candidate], gap: float = 3.0) -> list[Candidate]:
    """Одинаковый тип в пределах gap секунд — одно событие (лучшее по уверенности)."""
    out: list[Candidate] = []
    for c in sorted(cands, key=lambda c: (c.type_id, c.t_start)):
        if out and out[-1].type_id == c.type_id and c.t_start - out[-1].t_end <= gap:
            prev = out[-1]
            keep = c if c.confidence > prev.confidence else prev
            keep.t_start, keep.t_end = min(prev.t_start, c.t_start), max(prev.t_end, c.t_end)
            out[-1] = keep
        else:
            out.append(c)
    return sorted(out, key=lambda c: c.t_peak)


def _evidence(video_path, c):
    """1–3 кадра; если отрисовка рамок упала — хотя бы кадр на таймкоде (формат ТЗ требует ≥1)."""
    try:
        ev = make_evidence(video_path, c.evidence_ts or [c.t_peak], c.boxes, blur=config.BLUR_FACES)
        if ev:
            return ev
    except Exception as e:  # noqa: BLE001
        print(f"[pipeline] кадры-доказательства: {e}")
    return make_evidence(video_path, [c.t_peak], [], blur=config.BLUR_FACES)


def analyze(video_path: str | Path, with_frames: bool = True, progress=None) -> dict:
    t0 = time.time()
    video_path = str(video_path)
    meta = probe(video_path)
    perception, detectors, verifier = _perception(), build_detectors(), build_verifier()
    getattr(perception, "reset", lambda: None)()
    for d in detectors:
        d.reset()

    # 1. Покадровое восприятие (с контролем времени — лимит жюри 5 минут на видео)
    timeline, truncated = [], False
    # длинное видео — реже кадры, чтобы уложиться в лимит жюри (5 мин); MAX_FRAMES подбирается под железо
    import os
    max_frames = int(os.getenv("MAX_FRAMES", "1500"))
    fps = min(config.SAMPLE_FPS, max(2.0, max_frames / max(meta.duration_sec, 1.0)))
    for t, frame in iter_frames(video_path, fps, config.MAX_SIDE):
        obs = perception(t, frame)
        timeline.append(obs)
        for d in detectors:
            d.update(obs, frame)
        if progress and meta.duration_sec:
            progress(min(0.8, 0.8 * t / meta.duration_sec))
        if time.time() - t0 > config.TIME_BUDGET_SEC * 0.7:
            truncated = True
            break

    # 2. Rule-детекторы -> кандидаты (только включённые в regulations.yaml типы)
    types = rules.load_types()
    cands = []
    for d in detectors:   # сбой одного правила не должен ронять весь анализ
        try:
            cands += [c for c in d.finalize(timeline) if types.get(c.type_id) and types[c.type_id].enabled]
        except Exception as e:  # noqa: BLE001
            print(f"[pipeline] правило {type(d).__name__} упало: {e}")
    cands = _dedup(cands)

    # 3. Верификация (VLM), пока есть время
    confirmed = []
    for c in cands:
        if verifier and time.time() - t0 < config.TIME_BUDGET_SEC * 0.9:
            ok, conf, desc = verifier.verify(video_path, c)
            if not ok:
                continue
            c.confidence = conf
            c.observed = desc or c.observed
        confirmed.append(c)
    if progress:
        progress(0.9)

    # 4. Сборка ответа
    violations = []
    for c in confirmed:
        vt = rules.get(c.type_id)
        violations.append({
            "type": vt.name,
            "timecode_sec": round(float(c.t_peak), 2),
            "description": rules.build_description(vt, c.observed),
            "evidence_frames": _evidence(video_path, c) if with_frames else [],
            "rule": f"{vt.clause}: {vt.quote}",
            "source": f"{vt.source_act}. {vt.source_title}. {vt.source_url}",
            "confidence": round(float(c.confidence), 3),
        })

    return {
        "verdict": "violation" if violations else "no_violation",
        "violations": violations,
        "_meta": {"duration_sec": round(meta.duration_sec, 2), "fps": meta.fps, "frames_analyzed": len(timeline),
                  "processing_sec": round(time.time() - t0, 2), "truncated": truncated},
    }


if __name__ == "__main__":
    res = analyze(sys.argv[1], with_frames="--frames" in sys.argv)
    print(json.dumps(res, ensure_ascii=False, indent=2))
