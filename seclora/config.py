"""Validated configuration for the SecLoRA backend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class SecLoRAConfig:
    mode: str
    ratio: float
    sfp: int
    xmax: float
    threads: int

    @property
    def scale(self) -> int:
        return 1 << self.sfp

    @classmethod
    def from_root(cls, root: dict) -> Optional["SecLoRAConfig"]:
        raw = root.get("secure_aggregation")
        if raw is None or raw.get("backend", "plaintext") == "plaintext":
            return None
        if raw.get("backend") != "seclora":
            raise ValueError(
                f"Unsupported secure_aggregation.backend: {raw.get('backend')}"
            )

        config = cls(
            mode=str(raw.get("mode", "sel-2s")).lower(),
            ratio=float(raw.get("ratio", 0.1)),
            sfp=int(raw.get("sfp", 22)),
            xmax=float(raw.get("xmax", 0.03125)),
            threads=int(raw.get("threads", 1)),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.mode != "sel-2s":
            raise ValueError(
                "The first end-to-end milestone supports mode=sel-2s only"
            )
        if not 0.0 <= self.ratio < 1.0:
            raise ValueError("secure_aggregation.ratio must be in [0, 1)")
        if not 1 <= self.sfp <= 30:
            raise ValueError("secure_aggregation.sfp must be in [1, 30]")
        if self.xmax <= 0.0:
            raise ValueError("secure_aggregation.xmax must be positive")
        if self.threads <= 0:
            raise ValueError("secure_aggregation.threads must be positive")
