# CanaryMorph

AI voice timbre transfer based on **RVC v2**. Train a model from your reference voice samples, then convert any other voice to that timbre while preserving the source's words, prosody and rhythm.

Output is always **48000 Hz, 24-bit PCM, mono WAV**.

---

## Requirements

- Node.js ≥ 20.11
- pnpm ≥ 9
- Python 3.10.x (RVC is incompatible with 3.12+)
- ffmpeg — bundled via `ffmpeg-static`, no system install needed
- GPU is optional but ~10–20× faster than CPU/MPS for training:
  - **NVIDIA**: CUDA 12.1 wheels installed automatically
  - **Apple Silicon**: PyTorch MPS works for inference; training works but is slower and may NaN — CPU is the safer default on Mac
  - **CPU**: works everywhere, slow

---

## Quick start

```bash
pnpm install
pnpm run setup                       # downloads ~880 MB of pretrained weights
# drop your reference voice files into ./source/
pnpm train
pnpm transform input.mp3 result.wav
```

---

## Commands

### `pnpm setup`

Creates the Python virtualenv, installs pinned ML dependencies, downloads `hubert_base.pt`, `rmvpe.pt`, and the 48 kHz v2 pretrained generator/discriminator into `models/pretrained/`. Idempotent — re-running skips files that are already present.

### `pnpm train [options]`

Reads everything in `./source/` (`.wav .mp3 .flac .m4a .ogg`), slices it into training chunks, extracts pitch and HuBERT features, fine-tunes the pretrained 48k v2 generator, and builds a Faiss retrieval index.

Outputs:

- `models/trained/canary.pth` — generator weights
- `models/trained/canary.index` — Faiss feature index (critical for quality)

### `pnpm transform <input> <output.wav> [options]`

Converts `<input>` (any audio format) to the trained timbre and writes a 48 kHz / 24-bit / mono WAV to `<output>`.

---

## Full flag reference

### Train flags

| Flag | Default | Range | What it does |
| --- | --- | --- | --- |
| `-e, --epochs <n>` | `200` | 50–1000 | Total training epochs. Too few → generic/robotic. Too many → metallic artifacts and overfitting on recording noise. |
| `-b, --batch-size <n>` | `4` | 1–16 | Samples per training step. Larger = faster on big GPUs but more VRAM. Use 8 only if you have >12 GB VRAM. |
| `--save-every <n>` | `50` | 5–100 | Checkpoint frequency. Lower = safer (more rollback points) but slower disk usage. |
| `--top-db <db>` | `30` | 20–50 | Silence threshold for slicing. **Higher = more aggressive trim**, denser training data, better timbre learning. Use 40 for clean studio audio, 25 for noisy recordings. |
| `--device <name>` | `auto` | `auto cpu mps cuda` | Force a backend. `auto` picks CUDA → MPS → CPU. On Mac, training on CPU avoids occasional MPS NaNs. |
| `--cache-in-gpu` | `false` | flag | Keep the entire dataset in GPU memory. CUDA only; needs >12 GB VRAM. ~30 % faster when it fits. |
| `--reprep` | `false` | flag | Re-run dataset preparation. Use whenever you change `./source/` or `--top-db`. |

### Transform flags

| Flag | Default | Range | What it does |
| --- | --- | --- | --- |
| `-p, --pitch <semitones>` | `0` | -24 to +24 | Pitch shift before conversion. Use `+12` for male→female, `-12` for female→male. |
| `--method <name>` | `rmvpe` | `rmvpe pm harvest crepe` | Pitch extractor. `rmvpe` is the highest-quality default. `crepe` is slightly more accurate on melodic content but slower. `pm` and `harvest` are legacy fallbacks. |
| `--index-rate <0-1>` | `0.75` | 0.0–1.0 | How strongly to retrieve reference timbre from the Faiss index. **The most important quality knob.** 0 = ignore reference, 1 = full retrieval. Above 0.85 starts adding artifacts on consonants. |
| `--protect <0-0.5>` | `0.33` | 0.0–0.5 | Protects unvoiced consonants (s, t, k, sh) from over-conversion. Lower = more reference timbre on consonants but more "lispy" or robotic artifacts. |
| `--rms-mix-rate <0-1>` | `0.25` | 0.0–1.0 | Blend the reference's loudness envelope. 0 = keep source dynamics (recommended for speech), 1 = match reference dynamics (better for sustained singing). |
| `--filter-radius <n>` | `3` | odd 0–7 | Median filter on extracted f0. Higher smooths pitch jitter but flattens expressive pitch. |
| `--device <name>` | `auto` | `auto cpu mps cuda` | Inference backend. |

---

## Recommended commands

### Maximum-quality training (recommended)

Assumes 10–30 minutes of clean, single-speaker reference audio in `./source/`.

```bash
pnpm train \
  --epochs 400 \
  --batch-size 4 \
  --save-every 25 \
  --top-db 40 \
  --reprep
```

Why these values:

- `--epochs 400` — RVC v2 keeps improving past the default 200; 400 is a sweet spot before overfitting kicks in. For 30 min+ of audio, drop to 250–300. For <10 min, stay at 200 to avoid overfitting.
- `--top-db 40` — aggressive silence trimming, gives the model only voiced frames. Drop to 25–30 if your recordings have audible background noise that gets cut as "speech".
- `--save-every 25` — more frequent checkpoints, in case a later epoch overfits and you need to roll back manually.
- `--reprep` — always include this on the first run after editing `./source/`.

If you have an NVIDIA GPU with >12 GB VRAM:

```bash
pnpm train --epochs 400 --batch-size 8 --save-every 25 --top-db 40 --cache-in-gpu --reprep
```

### Maximum-similarity transformation

```bash
pnpm transform input.wav output.wav \
  --method rmvpe \
  --index-rate 0.85 \
  --protect 0.20 \
  --rms-mix-rate 0.25 \
  --filter-radius 3
```

Why these values:

- `--index-rate 0.85` — biggest similarity boost. Past 0.85 the consonants start to swim.
- `--protect 0.20` — pushes more reference timbre onto consonants. If output sounds lispy, raise to 0.30.
- `--rms-mix-rate 0.25` — preserves the source's dynamics (loud/quiet contrast). For sung vocals raise to 0.5–0.75.
- `--method rmvpe` and `--filter-radius 3` — already optimal defaults; listed explicitly for clarity.

If gendered pitch shift is needed:

```bash
pnpm transform male_input.wav female_output.wav --pitch 12 --index-rate 0.85 --protect 0.20
```

### Tuning recipes

| Symptom | Try |
| --- | --- |
| Output sounds too generic / "not like reference" | Raise `--index-rate` to 0.85; lower `--protect` to 0.20; train more epochs |
| "Lispy", muffled, or distorted consonants | Raise `--protect` to 0.40; lower `--index-rate` to 0.70 |
| Pitch warbling / wobble on long notes | Raise `--filter-radius` to 5 |
| Output too loud/quiet vs source | Lower `--rms-mix-rate` to 0.0 (full source dynamics) |
| Robotic / metallic timbre | Overfitting — retrain with fewer `--epochs` (e.g. 150) |
| Output sounds like a different person each phrase | `./source/` likely has multiple speakers — RVC averages them. Use one speaker's voice only. |

---

## Source data tips

- **One speaker only.** RVC has no speaker separation; multiple voices average into a "blended" timbre.
- **5–30 minutes** of clean speech is the sweet spot. 1–3 minutes works but quality plateaus low; >1 hour gives diminishing returns.
- **No music, no reverb, no background noise.** RVC will faithfully learn the noise as part of the timbre.
- **Single recording environment.** Switching mics/rooms mid-dataset hurts more than helps.
- **Lossless or high-bitrate sources** (.wav, .flac, 320 kbps mp3). Heavily compressed audio bakes codec artifacts into the model.

---

## Pipeline at a glance

```
input.mp3
   ↓ ffmpeg → 16 kHz mono PCM
RVC inference (HuBERT features → retrieve from canary.index → generate at 48 kHz)
   ↓ float32 wav
ffmpeg → pcm_s24le, 48000 Hz, mono
   ↓
output.wav
```

Each stage is independent; collapsing them loses quality.
