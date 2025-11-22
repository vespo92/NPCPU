"""
NPCPU Endocrine System

Hormonal regulation for digital organisms.
Manages long-term state and behavioral modulation.
"""

from .hormones import (
    EndocrineSystem,
    Hormone,
    HormoneType,
    Gland,
    HormoneEffect,
    MoodState,
    DriveState
)

__all__ = [
    "EndocrineSystem",
    "Hormone",
    "HormoneType",
    "Gland",
    "HormoneEffect",
    "MoodState",
    "DriveState"
]
