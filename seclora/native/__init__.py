"""Lazy loader for the compiled SecLoRA native extension."""

from __future__ import annotations


def create_native_session(**kwargs):
    try:
        from seclora.native import _seclora_native
    except ImportError as exc:
        raise RuntimeError(
            "SecLoRA native extension is not built. "
            "Run seclora/native/build.sh in the active environment."
        ) from exc
    return _seclora_native.SelectiveTwoServerSession(**kwargs)
