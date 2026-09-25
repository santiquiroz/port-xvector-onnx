from __future__ import annotations

import numpy as np
import pytest

from reference.fbank import HOP_LENGTH, compute_fbank, normalize_sentence
from tests.speechbrain_golden import GOLDEN_PATH, golden_signal

GOLDEN_TOLERANCE = 1e-3
SPEECHBRAIN_TOP_DB = 80.0
DOUBLING_DB = 20.0 * np.log10(2.0)


def voiced_half() -> np.ndarray:
    signal = golden_signal()
    return signal[: len(signal) // 2]


def test_matches_speechbrain_golden_features() -> None:
    expected = np.load(GOLDEN_PATH)

    actual = normalize_sentence(compute_fbank(golden_signal()))

    assert actual.shape == expected.shape
    assert np.abs(actual - expected).max() < GOLDEN_TOLERANCE


@pytest.mark.parametrize("samples", [400, 16000, 16159, 48000])
def test_frame_count_follows_centered_stft(samples: int) -> None:
    audio = np.resize(voiced_half(), samples)

    assert compute_fbank(audio).shape == (1 + samples // HOP_LENGTH, 24)


def test_features_are_power_decibels() -> None:
    audio = voiced_half()

    shift = compute_fbank(2.0 * audio) - compute_fbank(audio)

    np.testing.assert_allclose(shift, DOUBLING_DB, atol=1e-3)


def test_silence_is_clipped_to_top_db_below_the_peak() -> None:
    feats = compute_fbank(golden_signal())

    assert feats.min() == pytest.approx(feats.max() - SPEECHBRAIN_TOP_DB, abs=1e-4)


@pytest.mark.parametrize("shape", [(32000, 2), (2, 32000), (4, 16000)])
def test_rejects_multichannel_audio(shape: tuple[int, int]) -> None:
    with pytest.raises(ValueError, match="mono"):
        compute_fbank(np.zeros(shape, np.float32))


def test_rejects_three_dimensional_audio() -> None:
    with pytest.raises(ValueError, match="mono"):
        compute_fbank(np.zeros((1, 16000, 1), np.float32))


def test_column_vector_matches_flat_audio() -> None:
    audio = voiced_half()

    np.testing.assert_array_equal(compute_fbank(audio[:, None]), compute_fbank(audio))


def test_row_vector_matches_flat_audio() -> None:
    audio = voiced_half()

    np.testing.assert_array_equal(compute_fbank(audio[None, :]), compute_fbank(audio))


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_rejects_non_finite_samples(bad_value: float) -> None:
    audio = voiced_half().copy()
    audio[100] = bad_value

    with pytest.raises(ValueError, match="finite"):
        compute_fbank(audio)
