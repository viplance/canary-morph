import os
import sys
from pathlib import Path
import librosa
import soundfile as sf
import numpy as np

def prepare_dataset(source_dir: Path, dataset_dir: Path, target_sr: int = 48000, top_db: int = 30) -> int:
    source_dir = Path(source_dir)
    dataset_dir = Path(dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # 1. Glob files
    extensions = ['*.wav', '*.mp3', '*.flac', '*.ogg', '*.m4a']
    files = []
    for ext in extensions:
        files.extend(list(source_dir.rglob(ext)))

    if not files:
        print(f"Error: No audio files found in {source_dir}")
        return 0

    print(f"Found {len(files)} files. Processing...")

    slice_count = 0
    total_duration = 0
    rejected_short = 0
    rejected_quiet = 0

    # RVC's training script crops segments of `segment_size` samples from each slice
    # (typically 17280 @ 48k = 0.36 s) and then re-slices internally. Slices below
    # ~2 s sometimes end up empty after RVC's own preprocessing, which crashes
    # train.py with "tensor size 0". Min 2.0 s gives RVC headroom.
    MIN_DUR_S = 2.0
    # Reject near-silent slices that pass the silence-split threshold but contain
    # only breath/room noise. -40 dBFS RMS is the floor for usable speech.
    MIN_RMS_DBFS = -40.0
    min_rms_linear = 10 ** (MIN_RMS_DBFS / 20)

    # Stream files one at a time instead of concatenating the whole corpus into
    # one numpy array (which doubles peak RAM during np.concatenate and pins the
    # decoded audio for the lifetime of the loop).
    for f in files:
        try:
            y, _ = librosa.load(f, sr=target_sr, mono=True)
        except Exception as e:
            print(f"Warning: Failed to load {f}: {e}")
            continue

        y, _ = librosa.effects.trim(y, top_db=top_db)
        intervals = librosa.effects.split(y, top_db=top_db, frame_length=2048, hop_length=512)

        for start, end in intervals:
            dur = (end - start) / target_sr
            if dur < MIN_DUR_S:
                rejected_short += 1
                continue

            chunk = y[start:end]

            rms = float(np.sqrt(np.mean(chunk ** 2)))
            if rms < min_rms_linear:
                rejected_quiet += 1
                continue

            peak = float(np.max(np.abs(chunk)))
            if peak > 0:
                chunk = chunk * (10 ** (-1 / 20)) / peak

            out_path = dataset_dir / f"{slice_count:05d}.wav"
            sf.write(out_path, chunk, target_sr, subtype='FLOAT')

            slice_count += 1
            total_duration += dur

        del y

    print(
        f"Finished. Created {slice_count} slices. Total duration: {total_duration:.2f}s. "
        f"Rejected: {rejected_short} too-short (<{MIN_DUR_S}s), {rejected_quiet} too-quiet (<{MIN_RMS_DBFS} dBFS RMS)."
    )
    
    if total_duration < 10:
        print("Error: Not enough data (less than 10s).")
        return -1
    elif total_duration < 60:
        print("Warning: Less than 60s of audio. Quality might be low.")

    return slice_count

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--sr", type=int, default=48000)
    parser.add_argument("--top-db", type=int, default=30, dest="top_db")
    args = parser.parse_args()

    prepare_dataset(Path(args.source), Path(args.dataset), args.sr, args.top_db)
