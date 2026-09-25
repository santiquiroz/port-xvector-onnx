"""Check a downloaded tdnn.onnx: integrity, signature and separating power.

    python verify.py tdnn.onnx [--golden-dir reference/golden]

Needs only numpy + onnxruntime. Exits non-zero on any failure -- a model that
loads but returns embeddings from the wrong space is the exact failure this
export exists to avoid, so "it ran" is not the bar.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
from reference.fbank import SAMPLE_RATE, compute_fbank, normalize_sentence  # noqa: E402

EMBEDDING_DIM = 512
MANIFEST_PATH = REPO_ROOT / "manifest.json"
GOLDEN_DIR = REPO_ROOT / "reference" / "golden"
GOLDEN_SUFFIX = ".embedding.npy"
DEFAULT_OUTPUT_SPEC = ("batch", 1, EMBEDDING_DIM)
# The dilation-3 TDNN layer reflect-pads 3 frames per side, so fewer than 4
# frames crash here exactly as they crash SpeechBrain itself.
MIN_FRAMES = 4
EXTREME_FRAMES = (MIN_FRAMES, 50, 3000)
BATCH_COSINE = 0.9999
RERUN_COSINE = 0.9999
GOLDEN_COSINE = 0.999


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def synthetic_voice(
    f0: float, formants: tuple[float, ...], seconds: float = 3.0, seed: int = 0
) -> np.ndarray:
    """Harmonic source shaped by formant resonances -- speech-shaped enough.

    Two takes of the same (f0, formants) pair must embed closer to each other
    than to a different pair; that is the whole point of the encoder.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(int(SAMPLE_RATE * seconds)) / SAMPLE_RATE
    signal = np.zeros_like(t)
    for harmonic in range(1, int(SAMPLE_RATE / 2 / f0)):
        frequency = f0 * harmonic
        gain = sum(1.0 / (1.0 + ((frequency - f) / 90.0) ** 2) for f in formants)
        signal += gain * np.sin(2 * np.pi * frequency * t + rng.uniform(0, 2 * np.pi))
    envelope = 0.6 + 0.4 * np.sin(2 * np.pi * 3.0 * t)
    signal = signal * envelope + 0.005 * rng.standard_normal(t.shape)
    return (signal / np.abs(signal).max()).astype(np.float32)


def features(audio: np.ndarray) -> np.ndarray:
    return normalize_sentence(compute_fbank(audio))


def run_embeddings(session, feats: np.ndarray) -> np.ndarray:
    feeds = {"feats": feats.astype(np.float32)}
    return np.asarray(session.run(["embedding"], feeds)[0])


def embed(session, audio: np.ndarray) -> np.ndarray:
    vector = run_embeddings(session, features(audio)[None, :, :]).reshape(-1)
    return vector / np.linalg.norm(vector)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.reshape(-1), b.reshape(-1)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def load_manifest(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def output_spec(manifest: dict | None) -> tuple:
    if manifest is None:
        return DEFAULT_OUTPUT_SPEC
    return tuple(manifest["files"]["tdnn.onnx"]["outputs"]["embedding"])


def expected_output_shape(spec: tuple, batch: int) -> tuple:
    return tuple(batch if dim == "batch" else dim for dim in spec)


def repeat_frames(feats: np.ndarray, frames: int) -> np.ndarray:
    return np.take(feats, np.arange(frames) % len(feats), axis=0)


def integrity_problems(model: Path, manifest: dict | None) -> list[str]:
    if manifest is None:
        return []
    expected = manifest["files"]["tdnn.onnx"]
    actual_sha = sha256_of(model)
    print(f"sha256   {actual_sha}")
    problems = []
    if actual_sha != expected["sha256"]:
        problems.append("sha256 does not match the manifest")
    if model.stat().st_size != expected["bytes"]:
        problems.append(f"size {model.stat().st_size} != manifest {expected['bytes']}")
    return problems


def signature_problems(session) -> list[str]:
    inputs = {i.name: i.shape for i in session.get_inputs()}
    outputs = {o.name: o.shape for o in session.get_outputs()}
    print(f"inputs   {inputs}")
    print(f"outputs  {outputs}")
    problems = []
    if "feats" not in inputs:
        problems.append("no 'feats' input")
    if "embedding" not in outputs:
        problems.append("no 'embedding' output")
    return problems


def separation_problems(
    low: np.ndarray, high: np.ndarray, low_again: np.ndarray
) -> list[str]:
    print(f"dim      {low.shape[0]}")
    problems = []
    if low.shape[0] != EMBEDDING_DIM:
        problems.append(f"embedding is {low.shape[0]}-d, expected {EMBEDDING_DIM}")
    if not np.all(np.isfinite(low)):
        problems.append("embedding is not finite")

    # Raw x-vectors sit in a narrow cone -- every raw cosine is ~1.0, which is
    # why real systems center before scoring. Centering here too, otherwise the
    # margin is too small to tell a working export from a collapsed one.
    centroid = np.mean([low, high, low_again], axis=0)
    centered = [v - centroid for v in (low, high, low_again)]
    centered = [v / np.linalg.norm(v) for v in centered]
    same = float(centered[0] @ centered[2])
    different = float(centered[0] @ centered[1])
    print(f"cos(same speaker, other length)   {same:+.4f}  (centered)")
    print(f"cos(different speaker)            {different:+.4f}  (centered)")
    # Two takes of one voice have to land closer than two different voices.
    # A collapsed export makes every direction identical and fails this.
    if same <= different:
        problems.append("does not separate speakers (same <= different)")
    return problems


def shape_problems(output: np.ndarray, spec: tuple, batch: int) -> list[str]:
    expected = expected_output_shape(spec, batch)
    if output.shape == expected:
        return []
    return [f"output shape {output.shape} != manifest {expected}"]


def batch_problems(
    session, first: np.ndarray, second: np.ndarray, spec: tuple
) -> list[str]:
    batched = run_embeddings(session, np.stack([first, second]))
    print(f"batch    {batched.shape}")
    problems = shape_problems(batched, spec, 2)
    if problems:
        return problems
    for row, feats in enumerate((first, second)):
        score = cosine(batched[row], run_embeddings(session, feats[None, :, :]))
        print(f"cos(batch row {row}, alone)           {score:.7f}")
        if score <= BATCH_COSINE:
            problems.append(f"batch row {row}: cosine {score:.4f} <= {BATCH_COSINE}")
    return problems


def first_line(text: str) -> str:
    return text.partition("\n")[0]


def single_length_problems(session, feats: np.ndarray) -> list[str]:
    frames = len(feats)
    try:
        output = run_embeddings(session, feats[None, :, :])
    except Exception as error:  # noqa: BLE001 -- any runtime failure is a verdict
        return [f"{frames} frames: {first_line(str(error))}"]
    print(f"frames   {frames:>4} -> {output.shape}")
    if not np.all(np.isfinite(output)):
        return [f"{frames} frames: embedding is not finite"]
    return []


def length_problems(session, feats: np.ndarray) -> list[str]:
    problems = []
    for frames in EXTREME_FRAMES:
        problems += single_length_problems(session, repeat_frames(feats, frames))
    return problems


def stability_problems(session, feats: np.ndarray) -> list[str]:
    first = run_embeddings(session, feats[None, :, :])
    second = run_embeddings(session, feats[None, :, :])
    score = cosine(first, second)
    print(f"cos(same input, two runs)         {score:.7f}")
    if score > RERUN_COSINE:
        return []
    return [f"two runs of one input: cosine {score:.4f} <= {RERUN_COSINE}"]


def golden_signal_problems(session, name: str, audio, golden: np.ndarray) -> list[str]:
    if audio is None:
        return [f"golden {name} has no matching signal in parity/fixtures.py"]
    score = cosine(embed(session, audio), golden)
    print(f"golden   {name:<11} cos {score:.7f}")
    if score > GOLDEN_COSINE:
        return []
    return [f"golden {name}: cosine {score:.4f} <= {GOLDEN_COSINE}"]


def golden_problems(session, golden_dir: Path) -> list[str]:
    paths = sorted(golden_dir.glob(f"*{GOLDEN_SUFFIX}"))
    if not paths:
        print(f"golden   none in {golden_dir}, skipped")
        return []
    # Imported here: parity.fixtures itself imports synthetic_voice from this module.
    from parity.fixtures import parity_signals

    signals = parity_signals()
    problems = []
    for path in paths:
        name = path.name.removesuffix(GOLDEN_SUFFIX)
        problems += golden_signal_problems(
            session, name, signals.get(name), np.load(path)
        )
    return problems


def model_problems(session, spec: tuple, golden_dir: Path) -> list[str]:
    deep = (110.0, (500.0, 1100.0, 2400.0))
    bright = (215.0, (800.0, 1900.0, 3100.0))
    deep_audio = synthetic_voice(*deep, seed=1)
    bright_audio = synthetic_voice(*bright, seed=2)
    low = embed(session, deep_audio)
    high = embed(session, bright_audio)
    low_again = embed(session, synthetic_voice(*deep, seconds=2.0, seed=3))
    deep_feats = features(deep_audio)
    return [
        *separation_problems(low, high, low_again),
        *batch_problems(session, deep_feats, features(bright_audio), spec),
        *length_problems(session, deep_feats),
        *stability_problems(session, deep_feats),
        *golden_problems(session, golden_dir),
    ]


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model", nargs="?", type=Path, default=Path("tdnn.onnx"))
    parser.add_argument("--golden-dir", type=Path, default=GOLDEN_DIR)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.model.exists():
        print(f"FAIL: {args.model} does not exist.")
        return 2

    manifest = load_manifest(MANIFEST_PATH)
    problems = integrity_problems(args.model, manifest)

    import onnxruntime as ort

    session = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
    problems += signature_problems(session)
    problems += model_problems(session, output_spec(manifest), args.golden_dir)

    if problems:
        print("\nFAIL: " + "; ".join(problems))
        return 1
    print(
        "\nOK: integrity, signature, speaker separation, dynamic shapes, "
        "stability and golden parity all check out."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
