from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from parity.check_frontend_parity import (
    FEATURE_TOLERANCE,
    MIN_COSINE,
    ParityRow,
    cosine,
)
from parity.fixtures import (
    GOLDEN_DIR,
    META_PATH,
    embedding_path,
    features_path,
    parity_signals,
)
from reference.fbank import compute_fbank, normalize_sentence

MODEL_PATH = Path(__file__).resolve().parents[1] / "tdnn.onnx"
GOLDEN_BUDGET_BYTES = 1 << 20
SIGNALS = parity_signals()


def passing_row(**overrides: float) -> ParityRow:
    fields = {
        "name": "voice_3s",
        "reference_frames": 301,
        "speechbrain_frames": 301,
        "max_abs_diff": 1e-4,
        "std_ratio": 1.0,
        "cosine": 0.99999,
    }
    return ParityRow(**{**fields, **overrides})


@pytest.fixture(scope="module")
def onnx_session():
    if not MODEL_PATH.exists():
        pytest.skip("tdnn.onnx is a release asset; download it to run this check")
    import onnxruntime as ort

    return ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])


@pytest.mark.parametrize("name", sorted(SIGNALS))
def test_features_match_speechbrain_golden(name: str) -> None:
    expected = np.load(features_path(name))

    actual = normalize_sentence(compute_fbank(SIGNALS[name]))

    assert actual.shape == expected.shape
    assert np.abs(actual - expected).max() < FEATURE_TOLERANCE


@pytest.mark.parametrize("name", sorted(SIGNALS))
def test_onnx_embedding_matches_speechbrain_golden(name: str, onnx_session) -> None:
    feats = normalize_sentence(compute_fbank(SIGNALS[name]))[None, :, :]
    actual = onnx_session.run(["embedding"], {"feats": feats})[0].reshape(-1)

    assert cosine(actual, np.load(embedding_path(name))) > MIN_COSINE


def test_golden_records_its_provenance() -> None:
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))

    assert meta["speechbrain"]
    assert meta["model"]["revision"]
    assert sorted(meta["signals"]) == sorted(SIGNALS)


def test_golden_stays_under_one_megabyte() -> None:
    total = sum(path.stat().st_size for path in GOLDEN_DIR.iterdir())

    assert total < GOLDEN_BUDGET_BYTES


def test_row_passes_within_tolerances() -> None:
    assert passing_row().passes()


@pytest.mark.parametrize(
    "overrides",
    [
        {"reference_frames": 298},
        {"max_abs_diff": FEATURE_TOLERANCE},
        {"cosine": MIN_COSINE},
    ],
    ids=["frame-count", "feature-diff", "cosine"],
)
def test_row_fails_outside_tolerances(overrides: dict[str, float]) -> None:
    assert not passing_row(**overrides).passes()
