# port-xvector-onnx

ONNX export of the **SpeechBrain x-vector speaker encoder**
([`speechbrain/spkrec-xvect-voxceleb`](https://huggingface.co/speechbrain/spkrec-xvect-voxceleb)),
so you can extract speaker embeddings with plain `onnxruntime` — no `torch`, no
`speechbrain`, no ~2 GB of dependencies.

**Release:** [`models-v1.1`](https://github.com/santiquiroz/port-xvector-onnx/releases/tag/models-v1.1) → `tdnn.onnx` (16.1 MB, the same graph as `models-v1.0`; the release fixes the documented frontend)

| | |
|---|---|
| Upstream | `speechbrain/spkrec-xvect-voxceleb` (TDNN + statistical pooling, trained on VoxCeleb 1+2) |
| Upstream license | Apache-2.0 |
| This export | Apache-2.0 (same license, see `NOTICE`) |
| Opset | 17 |
| Input | `feats` — `float32[batch, frames, 24]` log-mel filterbank (see the contract below) |
| Output | `embedding` — `float32[batch, 1, 512]` |
| Max abs. difference vs. the PyTorch original | `< 2e-4` (asserted by the exporter) |
| sha256 | `dfb4daeb9b0a9aa33b0993e35f22841b34718fef1cfdcded2c7ff75867ddc7f8` |
| size | 16 897 902 bytes |

## Why this exists

[SpeechT5 VC](https://huggingface.co/microsoft/speecht5_vc) only clones a voice
properly when the speaker embedding comes from *this* encoder's space. Measured
2026-08-05: with WavLM embeddings the converted audio ended up **closer to the
source voice than to the target** — worse than not converting at all.

And `speechbrain` cannot always be installed next to the app that needs it:
1.1.0 has a broken lazy import, and 1.0.2 conflicts with `torchaudio` 2.11.

The two third-party x-vector ONNX exports published on Hugging Face were both
tested on 2026-08-05 and neither works: `arneyjfs/...` fails to initialise the
session, and `t-neethesh/...` loads and runs but returns embeddings outside
SpeechT5's space (downstream conversion comes out as NaN, 52 s of audio for a
3 s phrase). Hence this export.

## The feature frontend is NOT in the graph — read this before using it

`torch.onnx` cannot trace SpeechBrain's `compute_STFT` (the classic exporter
breaks on the reshape; the dynamo path fails to convert the graph). Only the
TDNN is exported, so **the caller must compute the filterbank itself**, exactly
as `speechbrain.lobes.features.Fbank(n_mels=24)` does with every other default
(speechbrain 1.0.2). Get it wrong and the embedding lands outside the space,
silently.

```
sample rate   16000 Hz, mono
n_mels        24
n_fft         400
win_length    400 samples (25 ms), periodic Hamming window
hop_length    160 samples (10 ms)
framing       centered: 200 zeros padded on each side, 1 + len // 160 frames
spectrum      power, |rfft|^2
f_min / f_max 0 / 8000 Hz, mel = 2595 * log10(1 + f / 700)
filters       SpeechBrain triangles over linspace(0, 8000, 201): centered on
              each inner mel point, half-width = distance to the previous one
compression   10 * log10(max(mel, 1e-10)), then floored at (peak - 80 dB),
              the peak taken over the whole utterance
normalisation subtract the per-utterance mean; do NOT divide by the std
              (speechbrain: norm_type="sentence", std_norm=False)
```

That last line matters: dividing by the standard deviation as well moves the
embedding out of the space and the downstream cloning stops working.

`reference/fbank.py` in this repo is a dependency-free NumPy implementation of
exactly the above, and `verify.py` runs the whole thing end to end.

## Usage

```python
import numpy as np, onnxruntime as ort
from reference.fbank import compute_fbank, normalize_sentence   # or your own

audio = ...  # float32, mono, 16 kHz, at least 480 samples (4 frames)
feats = normalize_sentence(compute_fbank(audio))[None, :, :]

session = ort.InferenceSession("tdnn.onnx", providers=["CPUExecutionProvider"])
embedding = session.run(["embedding"], {"feats": feats})[0].reshape(-1)
embedding /= np.linalg.norm(embedding)      # 512-d, unit norm
```

`compute_fbank` takes mono audio shaped `(N,)`, `(N, 1)` or `(1, N)`. Any other
shape (stereo `(N, 2)` from `soundfile.read`, for instance) raises `ValueError`
instead of interleaving the channels: mix them down first, e.g.
`audio.mean(axis=1)`. NaN or infinite samples raise `ValueError` too.

`compute_fbank` accepts 400 samples (3 frames), as SpeechBrain's `Fbank` does,
but the network needs 4 frames: its dilation-3 layer reflect-pads 3 frames on
each side, so 3 frames fail in onnxruntime exactly as they fail in SpeechBrain
(`Padding size should be less than the corresponding input dimension`).

## Reproducing the export

`export_xvector_onnx.py` is the script that produced the released artifact. It
needs a throwaway environment — this is exactly the ~2 GB of dependencies the
release exists to spare you:

```bash
python -m venv .venv-xvect
.venv-xvect/Scripts/pip install torch==2.5.1 torchaudio==2.5.1 speechbrain==1.0.2 \
    onnx onnxscript onnxruntime requests soundfile numpy huggingface_hub==0.25.2
.venv-xvect/Scripts/python export_xvector_onnx.py
```

It refuses to emit an artifact whose output drifts more than `1e-3` from the
PyTorch original.

## Verification

`verify.py` needs only `onnxruntime` and `numpy`:

```bash
python verify.py tdnn.onnx
```

It checks the sha256, the input/output signature, and that a real speech-shaped
signal produces a finite, unit-normalisable 512-d embedding whose cosine
similarity separates two different synthetic speakers. The export was traced
with a single `(1, 200, 24)` input, so it also exercises the dynamic axes:

- a batch of 2 has the manifest's output shape `[batch, 1, 512]` and each row
  matches the same input run alone (cosine > 0.9999);
- 4 (the minimum, see Usage), 50 and 3000 frames give finite embeddings;
- two runs of the same input agree (cosine > 0.9999) despite the pooling noise
  SpeechBrain adds, which the graph keeps;
- every `reference/golden/*.embedding.npy` (SpeechBrain's own embedding of the
  parity signals) matches ours with cosine > 0.999. `--golden-dir` points it
  elsewhere; with no golden embeddings this check is skipped.

### Parity against SpeechBrain

`parity/check_frontend_parity.py` runs in the export environment and measures
`reference/fbank.py` + `tdnn.onnx` against SpeechBrain itself: features against
the classifier's own `compute_features` + `mean_var_norm`, and the embedding
against `EncoderClassifier.encode_batch`, on a synthetic voice and seeded noise
of 0.5, 3 and 10 s. It exits non-zero unless every signal has the same frame
count, a feature difference below `1e-3` and a cosine above `0.999`, and it
rewrites `reference/golden/` (SpeechBrain's outputs plus `meta.json` with the
speechbrain version and model revision) so `pytest` can check the same thing
without torch:

```bash
.venv-xvect/Scripts/python parity/check_frontend_parity.py tdnn.onnx
python -m pytest -q
```

**Passes** (2026-09-25, speechbrain 1.0.2, torch 2.5.1 CPU, model revision
`56895a2`):

```
signal      frames ours/sb   max |diff| std ratio      cosine
voice_0.5s        51/51     8.58e-05    1.0000   1.0000001  ok
voice_3s        301/301     7.25e-05    1.0000   1.0000001  ok
voice_10s     1001/1001     9.54e-05    1.0000   0.9999999  ok
noise_0.5s        51/51     2.72e-05    1.0000   1.0000000  ok
noise_3s        301/301     2.86e-05    1.0000   1.0000001  ok
noise_10s     1001/1001     3.15e-05    1.0000   1.0000001  ok
```

The frontend shipped with `models-v1.0` fails it: 3 frames short, a std ratio
of about 0.11 and a cosine of only 0.90–0.93.

Verified end to end on 2026-08-10 (Windows 11, onnxruntime 1.24.4,
transformers 4.57.6, CPU): x-vector from a male reading → SpeechT5 VC → HiFi-GAN
turned 3.30 s of female speech into 3.71 s of finite audio (peak 0.4988,
RMS 0.1018, spectral distance 0.740 from the source), in 7.95 s.

### Tests

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

The suite needs neither torch nor the network. It covers `reference/fbank.py`
(frame count and shape, the short/multichannel/non-finite guards, power
decibels and the `top_db` floor, zero column mean without dividing by the std,
invariance to a constant gain, no empty mel filter, SpeechBrain's golden
features) and `verify.py` against fake sessions. The tests that need
`tdnn.onnx` are skipped until the release asset sits next to this README.

## License and attribution

The exported weights are a derivative of `speechbrain/spkrec-xvect-voxceleb`,
released by the SpeechBrain team under **Apache-2.0**; this repository keeps the
same license. See `LICENSE` and `NOTICE`.

The model was trained on **VoxCeleb 1+2**, which is distributed by the Oxford VGG
group under its own terms (CC BY 4.0 for the annotations, with the audio sourced
from YouTube). The Apache-2.0 grant here covers the released weights and this
export; if your use case is sensitive to training-data provenance, check
VoxCeleb's terms yourself.

```bibtex
@misc{speechbrain,
  title={{SpeechBrain}: A General-Purpose Speech Toolkit},
  author={Mirco Ravanelli and Titouan Parcollet and Peter Plantinga and others},
  year={2021}, eprint={2106.04624}, archivePrefix={arXiv}
}
@inproceedings{DBLP:conf/odyssey/SnyderGMSPK18,
  author={David Snyder and Daniel Garcia-Romero and Alan McCree and Gregory Sell
          and Daniel Povey and Sanjeev Khudanpur},
  title={Spoken Language Recognition using X-vectors},
  booktitle={Odyssey 2018}, pages={105--111}, year={2018}
}
```

Used by [Upflow](https://github.com/santiquiroz/upflow) for its voice-conversion
lane, alongside [port-audiosr-onnx](https://github.com/santiquiroz/port-audiosr-onnx)
and [port-gmfss-onnx](https://github.com/santiquiroz/port-gmfss-onnx).
