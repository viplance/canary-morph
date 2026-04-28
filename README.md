# CanaryMorph

AI voice timbre transfer based on RVC v2.

## Quick Start

1. **Requirements**: 
   - Node.js 20+
   - Python 3.10
   - FFmpeg (automatically handled via `ffmpeg-static`)
   - NVIDIA GPU (recommended for training)

2. **Setup**:
   ```bash
   pnpm install
   pnpm run setup
   ```

3. **Training**:
   Place your reference voice samples (`.wav`, `.mp3`, `.flac`, etc.) into the `./source/` directory. Aim for 5–30 minutes of clean, isolated speech.
   ```bash
   pnpm train
   ```

4. **Transformation**:
   Convert any voice file to the trained timbre.
   ```bash
   pnpm transform <input_file> <output_file.wav>
   ```

## Options

- **Pitch Shift**: Use `-p <semitones>` to shift the pitch during transformation.
  ```bash
  pnpm transform input.mp3 output.wav -p 12
  ```

- **Training Epochs**: Change the number of training epochs (default is 200). 
  An epoch is one full pass of the AI through your training data. 
  - **Too few epochs**: The voice will sound generic or robotic (underfitting).
  - **Too many epochs**: The voice may develop "metallic" artifacts or start mimicking specific noise from the recordings (overfitting).

  **Recommendations based on your data:**
  - **5–10 minutes of audio**: Use **200–300 epochs**.
  - **15–30 minutes of audio**: Use **100–200 epochs**.
  - **1 hour+ of audio**: Use **50–100 epochs**.

  ```bash
  pnpm train --epochs 150
  ```

- **Re-prepare Dataset**: Use `--reprep` to force re-slicing and feature extraction. Use this if you added new files to `./source/` or if a previous training run failed during preprocessing.
  ```bash
  pnpm train --reprep
  ```

## Output Format
Always produces **48000 Hz, 24-bit PCM, mono WAV** for maximum quality.
