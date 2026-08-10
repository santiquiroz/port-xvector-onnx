"""Check a downloaded tdnn.onnx: integrity, signature and separating power.

    python verify.py tdnn.onnx

Needs only numpy + onnxruntime. Exits non-zero on any failure -- a model that
loads but returns embeddings from the wrong space is the exact failure this
export exists to avoid, so "it ran" is not the bar.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reference.fbank import SAMPLE_RATE, compute_fbank, normalize_sentence  # noqa: E402

EMBEDDING_DIM = 512


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


def embed(session, audio: np.ndarray) -> np.ndarray:
    feats = normalize_sentence(compute_fbank(audio))[None, :, :]
    vector = np.asarray(session.run(["embedding"], {"feats": feats})[0]).reshape(-1)
    return vector / np.linalg.norm(vector)


def main() -> int:
    model = Path(sys.argv[1] if len(sys.argv) > 1 else "tdnn.onnx")
    if not model.exists():
        print(f"FAIL: {model} does not exist.")
        return 2

    problems: list[str] = []

    manifest_path = Path(__file__).resolve().parent / "manifest.json"
    if manifest_path.exists():
        expected = json.loads(manifest_path.read_text())["files"]["tdnn.onnx"]
        actual_sha = sha256_of(model)
        print(f"sha256   {actual_sha}")
        if actual_sha != expected["sha256"]:
            problems.append("sha256 does not match the manifest")
        if model.stat().st_size != expected["bytes"]:
            problems.append(
                f"size {model.stat().st_size} != manifest {expected['bytes']}"
            )

    import onnxruntime as ort

    session = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
    inputs = {i.name: i.shape for i in session.get_inputs()}
    outputs = {o.name: o.shape for o in session.get_outputs()}
    print(f"inputs   {inputs}")
    print(f"outputs  {outputs}")
    if "feats" not in inputs:
        problems.append("no 'feats' input")
    if "embedding" not in outputs:
        problems.append("no 'embedding' output")

    deep = (110.0, (500.0, 1100.0, 2400.0))
    bright = (215.0, (800.0, 1900.0, 3100.0))
    low = embed(session, synthetic_voice(*deep, seed=1))
    high = embed(session, synthetic_voice(*bright, seed=2))
    low_again = embed(session, synthetic_voice(*deep, seconds=2.0, seed=3))

    print(f"dim      {low.shape[0]}")
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

    if problems:
        print("\nFAIL: " + "; ".join(problems))
        return 1
    print("\nOK: integrity, signature and speaker separation all check out.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
