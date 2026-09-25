"""Dependency-free NumPy reimplementation of SpeechBrain's feature frontend.

The exported graph contains ONLY the TDNN: torch.onnx cannot trace SpeechBrain's
compute_STFT. So the filterbank has to be computed by the caller, and it has to
match bit-for-bit in spirit or the embedding lands outside the space -- silently,
with no error, producing garbage downstream.

It mirrors speechbrain.lobes.features.Fbank(n_mels=24) with every other default
(speechbrain 1.0.2), checked against it by tests/test_fbank.py.
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
# SpeechBrain's Filterbank: 10 * log10(max(power, AMIN)), then every value is
# raised to at least (utterance peak - TOP_DB).
AMIN = 1e-10
TOP_DB = 80.0


def _hz_to_mel(hz: np.ndarray | float) -> Any:
    return 2595.0 * np.log10(1.0 + np.asarray(hz) / 700.0)


def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def _mel_filterbank() -> np.ndarray:
    """HTK-formula mel triangles, built like Filterbank._create_fbank_matrix."""
    mel_points = np.linspace(_hz_to_mel(F_MIN), _hz_to_mel(F_MAX), N_MELS + 2)
    hz_points = _mel_to_hz(mel_points)
    centers = hz_points[1:-1, None]
    # SpeechBrain gives each triangle the width of its LEFT band on both sides.
    bands = (hz_points[1:-1] - hz_points[:-2])[:, None]
    freqs = np.linspace(0.0, SAMPLE_RATE / 2, N_FFT // 2 + 1)[None, :]
    slope = (freqs - centers) / bands
    return np.maximum(0.0, np.minimum(slope + 1.0, 1.0 - slope)).astype(np.float32)


_FILTERBANK = _mel_filterbank()


def _periodic_hamming(length: int) -> np.ndarray:
    """torch.hamming_window's default (periodic), not np.hamming (symmetric)."""
    return 0.54 - 0.46 * np.cos(2.0 * np.pi * np.arange(length) / length)


_WINDOW = _periodic_hamming(WIN_LENGTH)


def _centered_frames(audio: np.ndarray) -> np.ndarray:
    """Frames of torch.stft(center=True, pad_mode="constant")."""
    padded = np.pad(audio.astype(np.float64), N_FFT // 2)
    windows = np.lib.stride_tricks.sliding_window_view(padded, N_FFT)
    return windows[::HOP_LENGTH]


def _power_spectrum(frames: np.ndarray) -> np.ndarray:
    return np.abs(np.fft.rfft(frames * _WINDOW, n=N_FFT, axis=-1)) ** 2


def _to_decibels(mel: np.ndarray) -> np.ndarray:
    decibels = 10.0 * np.log10(np.maximum(mel, AMIN))
    return np.maximum(decibels, decibels.max() - TOP_DB)


def compute_fbank(audio: np.ndarray) -> np.ndarray:
    """Log-mel of the signal, shaped (frames, n_mels) as the TDNN expects."""
    if audio.ndim != 1:
        audio = audio.reshape(-1)
    if len(audio) < WIN_LENGTH:
        raise ValueError(
            f"Recording too short: at least {WIN_LENGTH / SAMPLE_RATE:.2f} s needed."
        )

    mel = _power_spectrum(_centered_frames(audio)) @ _FILTERBANK.T
    return _to_decibels(mel).astype(np.float32)


def normalize_sentence(feats: np.ndarray) -> np.ndarray:
    """Subtract the per-utterance mean. Do NOT divide by the std.

    SpeechBrain uses norm_type="sentence" with std_norm=False. Dividing by the
    standard deviation as well moves the embedding out of the space.
    """
    return feats - feats.mean(axis=0, keepdims=True)
