from __future__ import annotations

import itertools
import shutil
from pathlib import Path

import numpy as np
import pytest

import verify

MODEL_PATH = Path(__file__).resolve().parents[1] / "tdnn.onnx"
OUTPUT_SPEC = ("batch", 1, verify.EMBEDDING_DIM)
PROJECTION = np.random.default_rng(0).standard_normal((24, verify.EMBEDDING_DIM))
FEATS = np.random.default_rng(1).standard_normal((300, 24)).astype(np.float32)
OTHER_FEATS = np.random.default_rng(2).standard_normal((300, 24)).astype(np.float32)
NOISE_SEEDS = itertools.count(3)

requires_model = pytest.mark.skipif(
    not MODEL_PATH.exists(), reason="tdnn.onnx is a release asset; download it first"
)


class FakeSession:
    def __init__(self, embed_batch) -> None:
        self.embed_batch = embed_batch

    def run(self, names: list[str], feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
        return [self.embed_batch(feeds["feats"])]


def healthy_embedding(feats: np.ndarray) -> np.ndarray:
    return (feats.mean(axis=1) @ PROJECTION)[:, None, :]


def batch_mixing_embedding(feats: np.ndarray) -> np.ndarray:
    pooled = feats.mean(axis=1)
    return ((pooled + pooled.mean(axis=0)) @ PROJECTION)[:, None, :]


def noisy_embedding(feats: np.ndarray) -> np.ndarray:
    noise = np.random.default_rng(next(NOISE_SEEDS)).standard_normal(
        verify.EMBEDDING_DIM
    )
    return healthy_embedding(feats) + 100.0 * noise


def nan_on_long_input(feats: np.ndarray) -> np.ndarray:
    embedding = healthy_embedding(feats)
    return embedding * np.nan if feats.shape[1] > 1000 else embedding


def crash_on_short_input(feats: np.ndarray) -> np.ndarray:
    if feats.shape[1] < 10:
        raise RuntimeError("Pad reflect: pre-pad exceeds maximum allowed")
    return healthy_embedding(feats)


def flat_embedding(feats: np.ndarray) -> np.ndarray:
    return healthy_embedding(feats)[:, 0, :]


def altered_golden_dir(tmp_path: Path) -> Path:
    golden = tmp_path / "golden"
    shutil.copytree(verify.GOLDEN_DIR, golden)
    target = golden / "voice_3s.embedding.npy"
    np.save(target, -np.load(target))
    return golden


def test_expected_output_shape_binds_the_batch_axis() -> None:
    assert verify.expected_output_shape(OUTPUT_SPEC, 2) == (2, 1, 512)


def test_output_spec_comes_from_the_manifest() -> None:
    manifest = verify.load_manifest(verify.MANIFEST_PATH)

    assert verify.output_spec(manifest) == OUTPUT_SPEC


def test_output_spec_falls_back_without_manifest() -> None:
    assert verify.output_spec(None) == OUTPUT_SPEC


@pytest.mark.parametrize("frames", verify.EXTREME_FRAMES)
def test_repeat_frames_reaches_any_length(frames: int) -> None:
    assert verify.repeat_frames(FEATS, frames).shape == (frames, 24)


def test_shape_problems_accepts_the_manifest_shape() -> None:
    assert verify.shape_problems(np.zeros((2, 1, 512)), OUTPUT_SPEC, 2) == []


def test_shape_problems_rejects_a_flattened_output() -> None:
    assert verify.shape_problems(np.zeros((2, 512)), OUTPUT_SPEC, 2)


def test_batch_problems_accepts_independent_rows() -> None:
    session = FakeSession(healthy_embedding)

    assert verify.batch_problems(session, FEATS, OTHER_FEATS, OUTPUT_SPEC) == []


def test_batch_problems_catches_rows_that_leak_into_each_other() -> None:
    session = FakeSession(batch_mixing_embedding)

    assert verify.batch_problems(session, FEATS, OTHER_FEATS, OUTPUT_SPEC)


def test_batch_problems_catches_a_wrong_output_shape() -> None:
    session = FakeSession(flat_embedding)

    assert verify.batch_problems(session, FEATS, OTHER_FEATS, OUTPUT_SPEC)


def test_length_problems_accepts_finite_outputs() -> None:
    assert verify.length_problems(FakeSession(healthy_embedding), FEATS) == []


def test_length_problems_catches_non_finite_long_inputs() -> None:
    problems = verify.length_problems(FakeSession(nan_on_long_input), FEATS)

    assert any("3000 frames" in problem for problem in problems)


def test_length_problems_reports_a_crash_instead_of_raising() -> None:
    problems = verify.length_problems(FakeSession(crash_on_short_input), FEATS)

    assert any(f"{verify.MIN_FRAMES} frames" in problem for problem in problems)


def test_stability_problems_accepts_a_repeatable_model() -> None:
    assert verify.stability_problems(FakeSession(healthy_embedding), FEATS) == []


def test_stability_problems_catches_a_drifting_model() -> None:
    assert verify.stability_problems(FakeSession(noisy_embedding), FEATS)


def test_golden_check_is_skipped_without_fixtures(tmp_path: Path) -> None:
    session = FakeSession(healthy_embedding)

    assert verify.golden_problems(session, tmp_path) == []


def test_golden_check_rejects_an_embedding_outside_the_space() -> None:
    problems = verify.golden_problems(FakeSession(healthy_embedding), verify.GOLDEN_DIR)

    assert len(problems) == len(list(verify.GOLDEN_DIR.glob("*.embedding.npy")))


def test_missing_model_exits_with_two(tmp_path: Path) -> None:
    assert verify.main([str(tmp_path / "missing.onnx")]) == 2


@requires_model
def test_release_model_passes() -> None:
    assert verify.main([str(MODEL_PATH)]) == 0


@requires_model
def test_altered_golden_embedding_fails(tmp_path: Path) -> None:
    golden = altered_golden_dir(tmp_path)

    assert verify.main([str(MODEL_PATH), "--golden-dir", str(golden)]) == 1
