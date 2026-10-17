"""Справочник нарушений и сборка нормативной ссылки (только из configs/regulations.yaml)."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import yaml

from src.config import REGULATIONS


@dataclass
class ViolationType:
    id: str
    name: str
    enabled: bool
    source_title: str
    source_act: str
    source_url: str
    clause: str
    quote: str
    observable: str
    verified: bool

    def reference(self) -> str:
        return f"{self.source_act}, {self.clause}"


@lru_cache(maxsize=1)
def load_types() -> dict[str, ViolationType]:
    cfg = yaml.safe_load(REGULATIONS.read_text(encoding="utf-8"))
    out = {}
    for vid, v in cfg["violation_types"].items():
        src = cfg["sources"][v["source"]]
        out[vid] = ViolationType(vid, v["name"], bool(v.get("enabled", False)), src["title"].strip(),
                                 src["act"], src["url"], v["clause"], v["quote"], v["observable"],
                                 bool(v.get("verified", False)))
    return out


def get(vid: str) -> ViolationType:
    types = load_types()
    if vid not in types:
        raise KeyError(f"Тип нарушения '{vid}' отсутствует в {REGULATIONS.name} — сначала добавьте его туда")
    return types[vid]


def build_description(vt: ViolationType, observed: str) -> str:
    """observed — что именно видно (из детектора/VLM). Норматив добавляется из справочника."""
    return (f"{observed.rstrip('.')}. Это может указывать на нарушение: «{vt.quote}» "
            f"({vt.reference()}). Требуется проверка человеком.")
