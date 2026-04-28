import os
import sys
from pathlib import Path
import librosa
import soundfile as sf
import numpy as np

def prepare_dataset(source_dir: Path, dataset_dir: Path, target_sr: int = 48000) -> int:
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

    # 2. Load and concatenate
    all_audio = []
    for f in files:
        try:
            y, _ = librosa.load(f, sr=target_sr, mono=True)
            all_audio.append(y)
        except Exception as e:
            print(f"Warning: Failed to load {f}: {e}")

    if not all_audio:
        return 0

    y = np.concatenate(all_audio)

    # 4. Trim silence
    y, _ = librosa.effects.trim(y, top_db=30)

    # 5. Slice on silence
    # We use librosa.effects.split
    # RVC likes 3-10s slices.
    intervals = librosa.effects.split(y, top_db=30, frame_length=2048, hop_length=512)
    
    slice_count = 0
    total_duration = 0
    
    for start, end in intervals:
        dur = (end - start) / target_sr
        if dur < 1.0:
            continue
        
        # If longer than 15s, we should ideally split further, but for now we take it
        # as RVC's training script might handle it or we can sub-slice.
        # Minimal implementation as per plan:
        chunk = y[start:end]
        
        # 6. Peak normalize to -1 dBFS
        peak = np.max(np.abs(chunk))
        if peak > 0:
            chunk = chunk * (10**(-1/20)) / peak

        # 7. Save slice
        out_path = dataset_dir / f"{slice_count:05d}.wav"
        sf.write(out_path, chunk, target_sr, subtype='FLOAT')
        
        slice_count += 1
        total_duration += dur

    print(f"Finished. Created {slice_count} slices. Total duration: {total_duration:.2f}s")
    
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
    args = parser.parse_args()
    
    prepare_dataset(Path(args.source), Path(args.dataset), args.sr)
