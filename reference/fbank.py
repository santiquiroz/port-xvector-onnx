"""Dependency-free NumPy reimplementation of SpeechBrain's feature frontend.

The exported graph contains ONLY the TDNN: torch.onnx cannot trace SpeechBrain's
compute_STFT. So the filterbank has to be computed by the caller, and it has to
match bit-for-bit in spirit or the embedding lands outside the space -- silently,
with no error, producing garbage downstream.

Every constant here was read off the loaded PyTorch model, not from docs.
"""

from __future__ import annotations

from typing import Any

import numpy as np

SAMPLE_RATE = 16000
N_MELS = 24
N_FFT = 400
WIN_LENGTH = 400  # 25 ms at 16 kHz
HOP_LENGTH = 160  # 10 ms at 16 kHz
F_MIN = 0.0
F_MAX = 8000.0
# SpeechBrain takes the MAGNITUDE (power=0.5 over the power spectrum), not the
# power. Squaring here changes the log scale and shifts the embedding.
LOG_EPSILON = 1e-10


def _mel_filterbank() -> np.ndarray:
    """Slaney-scale mel filterbank, matching SpeechBrain's."""

    def hz_to_mel(hz: np.ndarray | float) -> Any:
        return 2595.0 * np.log10(1.0 + np.asarray(hz) / 700.0)

    def mel_to_hz(mel: np.ndarray) -> np.ndarray:
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    mel_points = np.linspace(hz_to_mel(F_MIN), hz_to_mel(F_MAX), N_MELS + 2)
    hz_points = mel_to_hz(mel_points)
    bins = np.floor((N_FFT + 1) * hz_points / SAMPLE_RATE).astype(int)

    bank = np.zeros((N_MELS, N_FFT // 2 + 1), dtype=np.float32)
    for m in range(1, N_MELS + 1):
        left, center, right = bins[m - 1], bins[m], bins[m + 1]
        for k in range(left, center):
            if center > left:
                bank[m - 1, k] = (k - left) / (center - left)
        for k in range(center, right):
            if right > center:
                bank[m - 1, k] = (right - k) / (right - center)
    return bank


_FILTERBANK = _mel_filterbank()


def compute_fbank(audio: np.ndarray) -> np.ndarray:
    """Log-mel of the signal, shaped (frames, n_mels) as the TDNN expects."""
    if audio.ndim != 1:
        audio = audio.reshape(-1)
    if len(audio) < WIN_LENGTH:
        raise ValueError(
            f"Recording too short: at least {WIN_LENGTH / SAMPLE_RATE:.2f} s needed."
        )

    window = np.hamming(WIN_LENGTH).astype(np.float32)
    frames = 1 + (len(audio) - WIN_LENGTH) // HOP_LENGTH

    spectrum = np.empty((frames, N_FFT // 2 + 1), dtype=np.float32)
    for i in range(frames):
        start = i * HOP_LENGTH
        chunk = audio[start : start + WIN_LENGTH] * window
        spectrum[i] = np.abs(np.fft.rfft(chunk, n=N_FFT))

    mel = spectrum @ _FILTERBANK.T
    return np.log(mel + LOG_EPSILON).astype(np.float32)


def normalize_sentence(feats: np.ndarray) -> np.ndarray:
    """Subtract the per-utterance mean. Do NOT divide by the std.

    SpeechBrain uses norm_type="sentence" with std_norm=False. Dividing by the
    standard deviation as well moves the embedding out of the space.
    """
    return feats - feats.mean(axis=0, keepdims=True)
