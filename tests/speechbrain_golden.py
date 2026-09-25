"""Golden features from SpeechBrain's own frontend, for tests/test_fbank.py.

Regenerate in the export environment (torch + speechbrain, see the README):

    .venv-xvect/Scripts/python tests/speechbrain_golden.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000
GOLDEN_PATH = Path(__file__).resolve().parent / "data" / "speechbrain_fbank.npy"


def golden_signal() -> np.ndarray:
    """One second of harmonics with formant-like gains, then exact silence.

    The silent half drives the power to zero, which exercises the amin floor
    and the top_db clip.
    """
    t = np.arange(SAMPLE_RATE // 2) / SAMPLE_RATE
    voiced = np.zeros_like(t)
    for harmonic in range(1, 60):
        frequency = 130.0 * harmonic
        gain = 1.0 / (1.0 + ((frequency - 700.0) / 150.0) ** 2) + 0.05
        voiced += gain * np.sin(2 * np.pi * frequency * t + 0.37 * harmonic)
    voiced *= 0.6 + 0.4 * np.sin(2 * np.pi * 3.0 * t)
    voiced /= np.abs(voiced).max()
    return np.concatenate([voiced, np.zeros_like(voiced)]).astype(np.float32)


def speechbrain_features(audio: np.ndarray) -> np.ndarray:
    import torch
    from speechbrain.lobes.features import Fbank
    from speechbrain.processing.features import InputNormalization

    compute_features = Fbank(n_mels=24).eval()
    mean_var_norm = InputNormalization(norm_type="sentence", std_norm=False).eval()
    with torch.no_grad():
        feats = compute_features(torch.from_numpy(audio)[None])
        return mean_var_norm(feats, torch.ones(1)).numpy()[0]


def main() -> None:
    GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(GOLDEN_PATH, speechbrain_features(golden_signal()).astype(np.float32))
    print(f"wrote {GOLDEN_PATH}")


if __name__ == "__main__":
    main()
