# CanaryMorph — Voice Timbre Transformation Implementation Plan

> **Audience:** Cheap AI coding agents (Haiku / GPT-4o-mini class).
> **Style:** Each step is small, deterministic, and independently verifiable.
> **Rule:** Do not skip steps. Do not "improve" the architecture. If a step fails, stop and report.

---

## 0. Goal

Build a CLI tool that:

1. **Trains** a voice-conversion model from reference voice samples in `./source/*.{wav,mp3}` (any speaker, any length, mono or stereo input — script normalizes).
2. **Transforms** any input voice file's timbre to match the trained reference, while preserving the input's linguistic content, prosody, and rhythm.
3. **Outputs** a `.wav` file at **48000 Hz, 24-bit PCM, mono**.

CLI usage:

```bash
pnpm train                                           # trains/refreshes model from ./source/
pnpm transform <input_voice.ext> <output_voice.wav>  # converts timbre
```

---

## 1. Architectural decision (READ — do not change)

We use **RVC (Retrieval-based Voice Conversion) v2** as the conversion engine. Reasons:

- State-of-the-art open-source timbre conversion as of 2025–2026.
- Trains usable models from **5–30 minutes** of clean reference audio (much less than VITS/SoVITS full training).
- Preserves source prosody/content; only transfers timbre.
- Mature Python tooling, MIT-licensed weights available.
- Faiss-indexed feature retrieval gives much higher fidelity than naive embedding conversion.

Alternative considered and **rejected**: `so-vits-svc` (better singing, but heavier training, worse for speech), `OpenVoice v2` (good zero-shot but lower fidelity than fine-tuned RVC), `XTTS-v2` (TTS, not voice conversion).

**Hybrid approach.** Node.js (TypeScript + pnpm) owns the CLI surface, file I/O, and audio I/O conversion. Python owns the ML pipeline (RVC). Node shells out to Python via a managed virtualenv. This keeps the CLI ergonomic while using the best ML stack.

---

## 2. Tech stack (pinned)

### Node side

- **Runtime:** Node.js ≥ 20.11
- **Package manager:** pnpm ≥ 9
- **Language:** TypeScript 5.5+ (strict mode)
- **CLI:** `commander` ^12
- **Process:** `execa` ^9
- **Logging:** `consola` ^3
- **Audio I/O & resampling:** `ffmpeg-static` ^5 + `fluent-ffmpeg` ^2 (final 48k/24-bit/mono encode)
- **Path/fs:** Node built-ins only

### Python side

- **Python:** 3.10.x (RVC-WebUI's tested version — do **not** use 3.12+)
- **PyTorch:** 2.2.x with CUDA 12.1 if NVIDIA GPU present, otherwise CPU build
- **RVC:** `rvc-python` (pip package wrapping Mangio/Applio fork) OR direct `Retrieval-based-Voice-Conversion-WebUI` clone — see Step 5
- **Pitch extractor:** `rmvpe` (most accurate, ships as a model file)
- **Feature extractor:** `ContentVec` (Hubert variant — ships as a model file)
- **Audio:** `librosa` 0.10.x, `soundfile` 0.12.x, `numpy` 1.26.x (do **not** use numpy 2.x — RVC code is incompatible)

---

## 3. Directory layout (create exactly this)

```
CanaryMorph/
├── package.json
├── pnpm-lock.yaml
├── tsconfig.json
├── .gitignore
├── .nvmrc                      # 20.11.1
├── README.md
├── implement.md                # this file
├── src/
│   ├── cli.ts                  # commander entry
│   ├── commands/
│   │   ├── train.ts
│   │   └── transform.ts
│   ├── lib/
│   │   ├── paths.ts            # resolves abs paths for models/source/python
│   │   ├── python.ts           # spawns python with the project venv
│   │   ├── ffmpeg.ts           # wav decode/encode helpers
│   │   └── logger.ts
│   └── types.ts
├── python/
│   ├── requirements.txt
│   ├── train.py                # called by `pnpm train`
│   ├── infer.py                # called by `pnpm transform`
│   └── utils/
│       ├── prep.py             # source audio cleanup + slicing
│       └── io.py               # numpy <-> wav helpers
├── source/                     # USER drops voice files here
│   └── .gitkeep
├── models/                     # generated; ignored in git
│   ├── pretrained/             # downloaded base RVC + ContentVec + RMVPE weights
│   ├── dataset/                # pre-processed slices for training
│   └── trained/
│       ├── canary.pth          # final RVC generator weights
│       └── canary.index        # faiss retrieval index
└── tmp/                        # ignored; intermediate files
```

`.gitignore` must include: `node_modules/`, `models/pretrained/`, `models/dataset/`, `models/trained/`, `tmp/`, `python/.venv/`, `*.log`.

---

## 4. Step-by-step implementation

> Each step is self-contained. **Run the verification command at the end of every step before moving on.** If verification fails, fix and re-verify.

### Step 1 — Bootstrap Node project

1. Create `package.json` with these exact scripts and deps:

```json
{
  "name": "canary-morph",
  "version": "0.1.0",
  "private": true,
  "type": "module",
  "engines": { "node": ">=20.11" },
  "scripts": {
    "build": "tsc",
    "setup": "tsx src/cli.ts setup",
    "train": "tsx src/cli.ts train",
    "transform": "tsx src/cli.ts transform"
  },
  "dependencies": {
    "commander": "^12.1.0",
    "consola": "^3.2.3",
    "execa": "^9.5.1",
    "ffmpeg-static": "^5.2.0",
    "fluent-ffmpeg": "^2.1.3"
  },
  "devDependencies": {
    "@types/fluent-ffmpeg": "^2.1.27",
    "@types/node": "^20.14.0",
    "tsx": "^4.19.0",
    "typescript": "^5.5.4"
  }
}
```

2. Create `tsconfig.json` with: `target: ES2022`, `module: ESNext`, `moduleResolution: Bundler`, `strict: true`, `noUncheckedIndexedAccess: true`, `outDir: dist`, `rootDir: src`, `esModuleInterop: true`, `skipLibCheck: true`, `resolveJsonModule: true`.

3. Create `.nvmrc` with `20.11.1`.

4. Run `pnpm install`.

**Verify:** `pnpm tsc --noEmit` exits 0.

---

### Step 2 — Implement `src/lib/paths.ts`

Resolve all absolute paths from the project root. Export:

```ts
export const ROOT: string; // dirname of package.json (use process.cwd() validated against package.json)
export const SOURCE_DIR: string; // ROOT/source
export const MODELS_DIR: string; // ROOT/models
export const PRETRAINED_DIR: string; // ROOT/models/pretrained
export const DATASET_DIR: string; // ROOT/models/dataset
export const TRAINED_DIR: string; // ROOT/models/trained
export const TMP_DIR: string; // ROOT/tmp
export const PYTHON_DIR: string; // ROOT/python
export const VENV_DIR: string; // ROOT/python/.venv
export const VENV_PYTHON: string; // platform-correct python in venv
export const MODEL_NAME = 'canary'; // constant
export const MODEL_PTH: string; // TRAINED_DIR/canary.pth
export const MODEL_INDEX: string; // TRAINED_DIR/canary.index
```

`VENV_PYTHON` resolves to `${VENV_DIR}/bin/python` on darwin/linux and `${VENV_DIR}/Scripts/python.exe` on win32.

**Verify:** add a temporary `console.log` of all exports, run `tsx src/lib/paths.ts`, confirm paths print and use forward slashes (or backslashes on Windows). Remove the console.log after.

---

### Step 3 — Implement `src/lib/logger.ts` and `src/lib/python.ts`

`logger.ts`: re-export `consola` with a tagged instance: `export const log = consola.withTag("canary")`.

`python.ts`: export `runPython(scriptRelPath: string, args: string[], opts?: { cwd?: string }): Promise<void>`. It must:

1. Verify `VENV_PYTHON` exists; if not, throw with message "Run `pnpm setup` first."
2. Build absolute path to `${PYTHON_DIR}/${scriptRelPath}`.
3. Spawn via `execa` with `stdio: "inherit"`, `env: { ...process.env, PYTHONUNBUFFERED: "1", PYTHONIOENCODING: "utf-8" }`.
4. Reject with the captured non-zero exit code on failure.

**Verify:** `pnpm tsc --noEmit` exits 0.

---

### Step 4 — Implement `src/lib/ffmpeg.ts`

Wrap `fluent-ffmpeg` with `ffmpeg-static` binary path. Export two functions:

```ts
// Decode any input to 16-bit PCM WAV at 16000 Hz mono — RVC's required input format.
export async function decodeForRVC(
  inputPath: string,
  outWavPath: string,
): Promise<void>;

// Encode RVC output (44100 Hz mono float wav) to final 48000 Hz, 24-bit PCM, mono.
export async function encodeFinal(
  inputPath: string,
  outputPath: string,
): Promise<void>;
```

Implementation hints:

- `decodeForRVC`: `.audioChannels(1).audioFrequency(16000).audioCodec("pcm_s16le").format("wav")`.
- `encodeFinal`: `.audioChannels(1).audioFrequency(48000).audioCodec("pcm_s24le").format("wav")`. Use a high-quality resampler: add `.outputOptions(["-af", "aresample=resampler=soxr:precision=33"])`.
- Both must throw on ffmpeg error and resolve on `end`.
- Set `ffmpeg.setFfmpegPath(ffmpegStatic as unknown as string)` once at module load.

**Verify:** Add a quick smoke check: from a Node REPL run `decodeForRVC` on any mp3/wav, then `encodeFinal` on the produced wav, confirm the final file's `ffprobe` reports `s24le, 48000 Hz, 1 channel`.

---

### Step 5 — Implement `pnpm setup` command

This bootstraps the Python environment and downloads pretrained weights. It is a separate command; `train` and `transform` will call it automatically only if the venv is missing.

Create `src/commands/setup.ts` with `export async function runSetup(): Promise<void>`. It must, in order:

1. Ensure all directories from Step 3 exist (`fs.mkdir({ recursive: true })`).
2. Verify `python3.10 --version` works; if not, fail with: "Install Python 3.10.x (e.g., `brew install python@3.10` or `pyenv install 3.10.14`)".
3. If `VENV_DIR` does not exist: `python3.10 -m venv ${VENV_DIR}`.
4. Upgrade pip: `${VENV_PYTHON} -m pip install --upgrade pip wheel setuptools`.
5. Install requirements: `${VENV_PYTHON} -m pip install -r ${PYTHON_DIR}/requirements.txt`.
6. Download pretrained weights into `PRETRAINED_DIR` (idempotent — skip if files exist with correct size). Files (Hugging Face mirrors, all permissively licensed):

   | File                                            | Source URL                                                                                 | Size    |
   | ----------------------------------------------- | ------------------------------------------------------------------------------------------ | ------- |
   | `hubert_base.pt` (ContentVec)                   | `https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt`           | ~360 MB |
   | `rmvpe.pt`                                      | `https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt`                 | ~180 MB |
   | `f0G48k.pth` (pretrained generator, 48k v2)     | `https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/f0G48k.pth` | ~150 MB |
   | `f0D48k.pth` (pretrained discriminator, 48k v2) | `https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/f0D48k.pth` | ~190 MB |

   Use Node's built-in `fetch` + stream-to-file. **Verify SHA-256 if a known hash is recorded** (see step 5b); otherwise verify content-length matches the response header on completion.

7. Print "Setup complete." with `consola.success`.

**Verify:** `pnpm setup` finishes without error; `ls models/pretrained/` shows all four files; `python/.venv/bin/python -c "import torch, librosa, soundfile, faiss, numpy; print(torch.__version__)"` runs cleanly.

---

### Step 5b — `python/requirements.txt`

Pin to known-good versions for RVC v2:

```
--extra-index-url https://download.pytorch.org/whl/cu121
torch==2.2.2
torchaudio==2.2.2
numpy==1.26.4
scipy==1.11.4
librosa==0.10.2
soundfile==0.12.1
numba==0.59.1
faiss-cpu==1.8.0
praat-parselmouth==0.4.3
pyworld==0.3.4
torchcrepe==0.0.23
fairseq==0.12.2
einops==0.8.0
local-attention==1.9.14
tqdm==4.66.4
ffmpeg-python==0.2.0
```

> Note: On macOS Apple Silicon, drop the `--extra-index-url` line and the CUDA suffix is ignored — pip falls back to the CPU/MPS wheel. Add a check in `setup.ts`: if `process.platform === "darwin"`, write a filtered copy to `tmp/requirements.darwin.txt` (without the CUDA index line) and install from that instead.

---

### Step 6 — Implement `python/utils/prep.py`

Purpose: turn the user's `./source/*.{wav,mp3,...}` files into clean training slices.

Function: `prepare_dataset(source_dir: Path, dataset_dir: Path, target_sr: int = 48000) -> int` — returns slice count.

Steps inside:

1. Recursively glob `source_dir` for `*.wav, *.mp3, *.flac, *.ogg, *.m4a`. Fail with a helpful message if zero files.
2. For each file, load with `librosa.load(path, sr=target_sr, mono=True)`.
3. Concatenate all audio (so very short files combine).
4. Trim leading/trailing silence with `librosa.effects.trim(top_db=30)`.
5. Slice on silence using `librosa.effects.split(y, top_db=30, frame_length=2048, hop_length=512)`. Reject slices shorter than 1.0 s or longer than 15.0 s (re-split long ones at the largest internal silence).
6. Loudness-normalize each slice to −23 LUFS (use `pyloudnorm` if available; otherwise peak-normalize to −1 dBFS — peak fallback is acceptable, do not add pyloudnorm to requirements unless stripping dynamics is desired).
7. Save each slice as `dataset_dir/<index>.wav`, 32-bit float WAV, target_sr Hz, mono, via `soundfile.write`.
8. Print final slice count and total duration.

If total duration < 60 s: warn but continue. If < 10 s: error out — not enough data.

---

### Step 7 — Implement `python/train.py`

CLI: `python train.py --dataset <dir> --pretrained <dir> --out <dir> --name canary --epochs 200 --batch-size 4 --sample-rate 48000`.

Use **`rvc-python`** package (`pip install rvc-python`) which exposes `from rvc_python.modules.train.train import train_model`. If that package version's API differs, fall back to invoking RVC's `train_nsf_sim_cache_sid_load_pretrain.py` directly via `subprocess` — clone `https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI` once into `models/rvc-src/` during `setup`.

Training flow inside `train.py`:

1. Parse args.
2. `prepare_dataset(...)` from Step 6 if `--dataset` is empty or `--reprep` flag is set.
3. **Feature extraction:**
   - For each slice: extract pitch with RMVPE (`models/pretrained/rmvpe.pt`) → save `.f0.npy` and `.f0nsf.npy`.
   - Extract ContentVec features with `hubert_base.pt` (12th-layer hidden states for v2) → save `.npy` per slice.
4. **Filelist generation:** write `dataset/filelist.txt` with one line per slice: `wav|f0|f0nsf|feat|0` (speaker id 0).
5. **Training:** initialize generator from `f0G48k.pth`, discriminator from `f0D48k.pth`. Train with:
   - `sample_rate=48000`, `if_f0=1`, `version=v2`, `gpus=auto`, `total_epoch=epochs`, `save_every_epoch=25`, `cache_in_gpu=true if cuda else false`, `pretrain_g`, `pretrain_d` set to the downloaded weights, `if_save_latest=1`, `if_save_every_weights=0`.
   - Optimizer: AdamW, lr=1e-4, betas=(0.8, 0.99). These are RVC v2 defaults — do not change.
6. **Index build (CRITICAL for quality):** after training, gather all per-slice ContentVec `.npy` files, stack, and build a Faiss `IndexIVFFlat` index with `nlist = max(1, total_features // 39)` and `nprobe = 1`. Save to `models/trained/canary.index`.
7. Copy the final epoch's generator `.pth` to `models/trained/canary.pth`.

Print clear progress every epoch: `epoch N/M loss_g=… loss_d=…`.

**Defaults you must use:** epochs=200, batch_size=4 (raise to 8 if >12 GB VRAM), sample_rate=48000, pitch_method=rmvpe.

---

### Step 8 — Implement `python/infer.py`

CLI: `python infer.py --input <path> --output <path> --model <pth> --index <index> --pitch 0 --method rmvpe`.

Flow:

1. Parse args. `--pitch` is semitone shift (default 0 — keep source pitch; expose to user via Node CLI later).
2. Load input WAV (which Node already decoded to 16k mono 16-bit PCM). Resample to 16000 Hz mono if not already.
3. Initialize RVC inference pipeline (`rvc_python.infer` or vendored `vc_infer_pipeline.py`) with:
   - `hubert_model = load_hubert("models/pretrained/hubert_base.pt")`
   - `net_g = synthesizer_loaded_from(model_pth)` → must match v2, sr=48000, f0=true
   - `pitch_extraction = "rmvpe"`, `rmvpe_model = "models/pretrained/rmvpe.pt"`
   - `index_path = canary.index`, `index_rate = 0.75` (high-quality timbre retrieval; do not raise above 0.85 — causes artifacts)
   - `filter_radius = 3` (median-filter the f0 — smooths pitch jitter)
   - `rms_mix_rate = 0.25` (preserves source dynamics)
   - `protect = 0.33` (protects unvoiced consonants from over-conversion — leave at default)
4. Run conversion. Output is float32 mono at 48000 Hz (because the loaded generator is the 48k variant).
5. Save with `soundfile.write(out, audio, 48000, subtype="FLOAT")` to a temp path. **Do not** write 24-bit here — Node's ffmpeg step does the final 24-bit encode using soxr precision=33 for cleaner quantization.

Print elapsed seconds.

---

### Step 9 — Implement `src/commands/train.ts`

Function `runTrain(opts: { epochs?: number; batchSize?: number; reprep?: boolean }): Promise<void>`:

1. Ensure setup is done; if `VENV_PYTHON` missing → call `runSetup()` first.
2. Verify `SOURCE_DIR` contains at least one supported audio file; else error: "Place voice samples in ./source/ (.wav/.mp3/.flac/.m4a/.ogg)".
3. `runPython("train.py", ["--dataset", DATASET_DIR, "--pretrained", PRETRAINED_DIR, "--out", TRAINED_DIR, "--name", MODEL_NAME, "--epochs", String(opts.epochs ?? 200), "--batch-size", String(opts.batchSize ?? 4), "--sample-rate", "48000", ...(opts.reprep ? ["--reprep"] : [])])`.
4. Verify `MODEL_PTH` and `MODEL_INDEX` now exist; else error.
5. Log success with model path.

---

### Step 10 — Implement `src/commands/transform.ts`

Function `runTransform(input: string, output: string, opts: { pitch?: number }): Promise<void>`:

1. Validate `input` exists and is a regular file.
2. Validate `output` ends with `.wav` (we always emit WAV — error if not).
3. Validate model exists: `MODEL_PTH` and `MODEL_INDEX`. If missing → error: "Train the model first: pnpm train".
4. Create unique tmp paths in `TMP_DIR`: `tmp/<uuid>.in16k.wav` and `tmp/<uuid>.out48k.wav`.
5. `await decodeForRVC(input, in16k)`.
6. `await runPython("infer.py", ["--input", in16k, "--output", out48kFloat, "--model", MODEL_PTH, "--index", MODEL_INDEX, "--pitch", String(opts.pitch ?? 0), "--method", "rmvpe"])`.
7. `await encodeFinal(out48kFloat, output)`. **This is the only step that touches the user's destination path** — guarantees 48000 Hz, 24-bit PCM, mono.
8. Delete tmp files in a `finally` block (do not delete `output` on error — let the user see partial state if encode failed).
9. Log: input duration, processing time, output path.

---

### Step 11 — Implement `src/cli.ts`

```ts
import { Command } from 'commander';
import { runSetup } from './commands/setup.js';
import { runTrain } from './commands/train.js';
import { runTransform } from './commands/transform.js';

const program = new Command();
program
  .name('canary')
  .description('Voice timbre transformation')
  .version('0.1.0');

program
  .command('setup')
  .description('Install Python deps and download pretrained models')
  .action(runSetup);

program
  .command('train')
  .option('-e, --epochs <n>', 'training epochs', (v) => parseInt(v, 10), 200)
  .option('-b, --batch-size <n>', 'batch size', (v) => parseInt(v, 10), 4)
  .option('--reprep', 're-run dataset preparation', false)
  .action((opts) => runTrain(opts));

program
  .command('transform <input> <output>')
  .option(
    '-p, --pitch <semitones>',
    'pitch shift in semitones',
    (v) => parseInt(v, 10),
    0,
  )
  .action((input, output, opts) => runTransform(input, output, opts));

program.parseAsync(process.argv).catch((err) => {
  console.error(err?.message ?? err);
  process.exit(1);
});
```

**Verify:** `pnpm tsc --noEmit` exits 0. `pnpm transform --help` prints usage.

---

### Step 12 — README.md

Short, command-focused. Sections: requirements (Node 20+, Python 3.10, ffmpeg auto-bundled, optional NVIDIA GPU), quick-start (`pnpm install` → drop files into `./source/` → `pnpm setup` → `pnpm train` → `pnpm transform in.mp3 out.wav`), tuning notes (epochs, pitch flag, source data quality tips: 5–30 min of clean speech is the sweet spot, no music/noise, single speaker).

---

## 5. Quality checklist (must all pass before declaring done)

Run each manually:

1. `pnpm install` succeeds.
2. `pnpm tsc --noEmit` succeeds.
3. `pnpm setup` completes; pretrained files present.
4. With ~10 minutes of clean reference audio in `./source/`: `pnpm train` completes 200 epochs without OOM (lower batch size if needed).
5. `pnpm transform sample.mp3 out.wav` produces `out.wav`.
6. `ffprobe out.wav` reports: `pcm_s24le`, `48000 Hz`, `mono` (1 channel). Non-negotiable.
7. Subjective listening: output is intelligible, retains source words and prosody, timbre clearly resembles reference.
8. Re-running `pnpm transform` with the same input is deterministic up to floating-point noise (RVC has no random sampling at inference).

---

## 6. Hard rules for agents

- **Pin every version.** Do not let pip/pnpm float.
- **Do not modify the audio path constants.** Input to RVC is 16k mono 16-bit. RVC output is 48k mono float. Final encode is 48k mono 24-bit. These are separate stages by design — collapsing them loses quality.
- **Do not skip Faiss index building.** Without `canary.index`, conversion quality drops drastically.
- **Do not change `index_rate`, `protect`, `rms_mix_rate`, or `filter_radius` defaults.** These are tuned values; reverting saves a future debug session.
- **Never download to system paths.** All weights live under `models/pretrained/`.
- **Never call `rm -rf` or delete user data outside `tmp/`.** Specifically, do not touch `source/`.
- **Do not add features not in this plan.** No web UI, no batch mode, no realtime. If the user asks later, that is a separate plan.
- **If a step fails:** report the exact command, exit code, and last 30 lines of stderr. Do not "try a different approach" without the user's go-ahead.

---

## 7. Known gotchas

- macOS Apple Silicon: PyTorch MPS backend works for inference but training is slower and occasionally NaNs. Recommend training on CPU on Mac (set `--gpus -` env var inside `train.py`) or on a Linux/CUDA box. Inference on MPS is fine.
- ffmpeg-static on Linux glibc < 2.31: ship-as-is fails. Document a fallback to system `ffmpeg` (env var `FFMPEG_PATH` overrides).
- fairseq install can fail on Python 3.11+ — that's why we pin 3.10.
- numpy 2.x breaks pyworld and parts of RVC — pinned to 1.26.4.
- Faiss-GPU is not on PyPI for macOS; use `faiss-cpu` everywhere — index build is fast enough on CPU.

---

## 8. Done definition

The plan is complete when:

- All 12 steps are implemented.
- All 8 quality checks pass.
- A user with no prior context can run `pnpm install && pnpm setup && pnpm train && pnpm transform a.mp3 b.wav` and get a 48 kHz / 24-bit / mono WAV that sounds like the reference voice speaking the input's words.
