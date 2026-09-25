"""Measure reference/fbank.py + tdnn.onnx against SpeechBrain itself.

Runs in the export environment (torch + speechbrain, see the README):

    .venv-xvect/Scripts/python parity/check_frontend_parity.py tdnn.onnx

For every deterministic signal in parity/fixtures.py it compares our features
with the classifier's own compute_features + mean_var_norm, and the ONNX
embedding with classifier.encode_batch. It rewrites reference/golden/ with
SpeechBrain's outputs, so tests/test_golden_parity.py can check the same thing
without torch. Exits non-zero if any signal is outside the tolerances.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from parity.fixtures import (
    GOLDEN_DIR,
    META_PATH,
    embedding_path,
    features_path,
    parity_signals,
)
from reference.fbank import compute_fbank, normalize_sentence
from verify import embed

FEATURE_TOLERANCE = 1e-3
MIN_COSINE = 0.999
MODEL_SOURCE = "speechbrain/spkrec-xvect-voxceleb"
DEFAULT_SAVEDIR = REPO_ROOT / ".tmp-xvect-src"
POOLING_NOISE_SEED = 0
HF_REPO_CACHE = "models--speechbrain--spkrec-xvect-voxceleb"
TOLERANCES = f"max |diff| < {FEATURE_TOLERANCE}, cosine > {MIN_COSINE}"
TABLE_HEADER = (
    f"{'signal':<11} {'frames ours/sb':>11} {'max |diff|':>12} "
    f"{'std ratio':>9} {'cosine':>11}"
)


@dataclass(frozen=True)
class ParityRow:
    name: str
    reference_frames: int
    speechbrain_frames: int
    max_abs_diff: float
    std_ratio: float
    cosine: float

    def passes(self) -> bool:
        return (
            self.reference_frames == self.speechbrain_frames
            and self.max_abs_diff < FEATURE_TOLERANCE
            and self.cosine > MIN_COSINE
        )


@dataclass(frozen=True)
class Measurement:
    row: ParityRow
    speechbrain_features: np.ndarray
    speechbrain_embedding: np.ndarray


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.reshape(-1), b.reshape(-1)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def max_abs_diff(ours: np.ndarray, theirs: np.ndarray) -> float:
    frames = min(len(ours), len(theirs))
    return float(np.abs(ours[:frames] - theirs[:frames]).max())


def std_ratio(ours: np.ndarray, theirs: np.ndarray) -> float:
    return float(ours.std() / theirs.std())


def load_classifier(savedir: Path):
    import speechbrain.utils.fetching as sb_fetching
    from speechbrain.inference.speaker import EncoderClassifier
    from speechbrain.utils.fetching import LocalStrategy

    # Same workaround as export_xvector_onnx.py: Windows without developer mode
    # cannot create the symlinks the pretrainer asks for, whatever strategy the
    # loader is given, so every fetch is forced to copy.
    link_original = sb_fetching.link_with_strategy

    def always_copy(src, dst, strategy):
        return link_original(src, dst, LocalStrategy.COPY)

    sb_fetching.link_with_strategy = always_copy
    return EncoderClassifier.from_hparams(
        source=MODEL_SOURCE, savedir=str(savedir), local_strategy=LocalStrategy.COPY
    )


def speechbrain_features(classifier, audio: np.ndarray) -> np.ndarray:
    import torch

    with torch.no_grad():
        feats = classifier.mods.compute_features(torch.from_numpy(audio)[None])
        return classifier.mods.mean_var_norm(feats, torch.ones(1)).numpy()[0]


def speechbrain_embedding(classifier, audio: np.ndarray) -> np.ndarray:
    import torch

    # StatisticsPooling adds eps-scale torch.randn noise to the pooled mean;
    # seeding it keeps the committed golden embeddings byte-stable across runs.
    torch.manual_seed(POOLING_NOISE_SEED)
    with torch.no_grad():
        return classifier.encode_batch(torch.from_numpy(audio)[None]).numpy()


def measure(classifier, session, name: str, audio: np.ndarray) -> Measurement:
    ours = normalize_sentence(compute_fbank(audio))
    theirs = speechbrain_features(classifier, audio)
    their_embedding = speechbrain_embedding(classifier, audio)
    row = ParityRow(
        name=name,
        reference_frames=len(ours),
        speechbrain_frames=len(theirs),
        max_abs_diff=max_abs_diff(ours, theirs),
        std_ratio=std_ratio(ours, theirs),
        cosine=cosine(embed(session, audio), their_embedding),
    )
    return Measurement(row, theirs, their_embedding)


def model_revision() -> str:
    from huggingface_hub import constants

    ref = Path(constants.HF_HUB_CACHE) / HF_REPO_CACHE / "refs" / "main"
    return ref.read_text().strip() if ref.exists() else "unknown"


def golden_meta(names: list[str]) -> dict:
    import speechbrain
    import torch

    return {
        "generator": "parity/check_frontend_parity.py",
        "model": {"source": MODEL_SOURCE, "revision": model_revision()},
        "speechbrain": speechbrain.__version__,
        "torch": torch.__version__,
        "features": "classifier.mods.compute_features + mean_var_norm, (frames, 24)",
        "embedding": "classifier.encode_batch, raw (1, 1, 512)",
        "signals": names,
    }


def write_golden(measurements: list[Measurement]) -> None:
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    for measurement in measurements:
        name = measurement.row.name
        np.save(
            features_path(name), measurement.speechbrain_features.astype(np.float32)
        )
        np.save(
            embedding_path(name), measurement.speechbrain_embedding.astype(np.float32)
        )
    meta = golden_meta([m.row.name for m in measurements])
    META_PATH.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


def format_row(row: ParityRow) -> str:
    verdict = "ok" if row.passes() else "FAIL"
    frames = f"{row.reference_frames}/{row.speechbrain_frames}"
    return (
        f"{row.name:<11} {frames:>11} {row.max_abs_diff:>12.2e} "
        f"{row.std_ratio:>9.4f} {row.cosine:>11.7f}  {verdict}"
    )


def format_table(rows: list[ParityRow]) -> str:
    return "\n".join([TABLE_HEADER, *(format_row(row) for row in rows)])


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model", type=Path, nargs="?", default=REPO_ROOT / "tdnn.onnx")
    parser.add_argument("--savedir", type=Path, default=DEFAULT_SAVEDIR)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.model.exists():
        print(f"FAIL: {args.model} does not exist.")
        return 2

    import onnxruntime as ort

    session = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
    classifier = load_classifier(args.savedir)
    measurements = [
        measure(classifier, session, name, audio)
        for name, audio in parity_signals().items()
    ]
    write_golden(measurements)

    rows = [m.row for m in measurements]
    print(format_table(rows))
    failed = [row.name for row in rows if not row.passes()]
    if failed:
        print(f"\nFAIL: outside {TOLERANCES}: {', '.join(failed)}")
        return 1
    print(f"\nOK: every signal within {TOLERANCES}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
