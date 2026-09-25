"""Deterministic parity signals and where their SpeechBrain golden outputs live."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from reference.fbank import SAMPLE_RATE
from verify import synthetic_voice

GOLDEN_DIR = Path(__file__).resolve().parents[1] / "reference" / "golden"
META_PATH = GOLDEN_DIR / "meta.json"

DURATIONS_S = (0.5, 3.0, 10.0)
VOICE = (110.0, (500.0, 1100.0, 2400.0))
VOICE_SEED = 0
NOISE_SEED = 7
NOISE_LEVEL = 0.1


def signal_name(kind: str, seconds: float) -> str:
    return f"{kind}_{seconds:g}s"


def seeded_noise(seconds: float) -> np.ndarray:
    rng = np.random.default_rng(NOISE_SEED)
    samples = int(SAMPLE_RATE * seconds)
    return (NOISE_LEVEL * rng.standard_normal(samples)).astype(np.float32)


def voice(seconds: float) -> np.ndarray:
    return synthetic_voice(*VOICE, seconds=seconds, seed=VOICE_SEED)


def parity_signals() -> dict[str, np.ndarray]:
    voices = {signal_name("voice", s): voice(s) for s in DURATIONS_S}
    noises = {signal_name("noise", s): seeded_noise(s) for s in DURATIONS_S}
    return {**voices, **noises}


def features_path(name: str) -> Path:
    return GOLDEN_DIR / f"{name}.feats.npy"


def embedding_path(name: str) -> Path:
    return GOLDEN_DIR / f"{name}.embedding.npy"
