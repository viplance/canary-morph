import os
import sys
import argparse
import subprocess
import json
import shutil
import numpy as np
from pathlib import Path
from utils.prep import prepare_dataset

def run_command(cmd, cwd=None):
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=cwd, check=True)

def train(
    source_dir: Path,
    dataset_dir: Path,
    pretrained_dir: Path,
    trained_dir: Path,
    name: str,
    epochs: int,
    batch_size: int,
    sample_rate: int,
    reprep: bool,
    save_every: int = 50,
    top_db: int = 30,
    device_pref: str = "auto",
    cache_in_gpu: bool = False,
):
    rvc_src = trained_dir.parent / "rvc-src"
    if not rvc_src.exists():
        print(f"Error: RVC source not found at {rvc_src}. Run setup first.")
        sys.exit(1)

    log_dir = rvc_src / "logs" / name
    log_dir.mkdir(parents=True, exist_ok=True)

    # 1. Pre-process dataset if needed
    if reprep or not (dataset_dir.exists() and any(dataset_dir.iterdir())):
        print(f"Preparing dataset slices from {source_dir} (top_db={top_db})...")
        count = prepare_dataset(source_dir, dataset_dir, sample_rate, top_db=top_db)
        if count <= 0:
            print("Failed to prepare dataset.")
            sys.exit(1)

    python_exe = sys.executable

    # Step A: RVC Preprocess
    run_command([
        python_exe, str(rvc_src / "infer/modules/train/preprocess.py"),
        str(dataset_dir), str(sample_rate), "2", str(log_dir), "False", "3.0"
    ], cwd=rvc_src)

    # Step B: Feature Extraction (F0)
    run_command([
        python_exe, str(rvc_src / "infer/modules/train/extract/extract_f0_print.py"),
        str(log_dir), "2", "rmvpe"
    ], cwd=rvc_src)

    # Step C: Feature Extraction (HuBERT)
    import torch  # local import to avoid forcing torch on prep-only invocations
    if device_pref == "cpu":
        device = "cpu"
    elif device_pref == "mps":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    elif device_pref == "cuda":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    else:  # auto
        device = "cpu"
        if torch.backends.mps.is_available():
            device = "mps"
        if torch.cuda.is_available():
            device = "cuda:0"
    print(f"Training device: {device}")
    
    run_command([
        python_exe, str(rvc_src / "infer/modules/train/extract_feature_print.py"),
        device, "1", "0", str(log_dir), "v2", "False"
    ], cwd=rvc_src)

    # Step D: Generate filelist.txt
    # CRITICAL: filter out slices whose feature length is below RVC's segment_size.
    # RVC's preprocess (preprocess.py) emits a "tail" chunk per source slice that
    # can be much shorter than the rest. If a slice's feature frames are fewer
    # than segment_size//hop_length, infer/lib/train/data_utils.py truncates the
    # paired wav to len_min*hop_length, which can collapse to 0 and crashes
    # commons.slice_segments with "Tensor sizes: [0]".
    gt_wavs_dir = log_dir / "0_gt_wavs"
    f0_dir = log_dir / "2a_f0"
    f0nsf_dir = log_dir / "2b-f0nsf"
    feature_dir = log_dir / "3_feature768"

    # Read hop_length, filter_length, and segment_size from the RVC config.
    cfg_for_seg = rvc_src / "configs" / "v2" / f"{sample_rate//1000}k.json"
    if not cfg_for_seg.exists():
        cfg_for_seg = rvc_src / "configs" / "v2" / "48k.json"
    with open(cfg_for_seg, "r") as f:
        _cfg = json.load(f)
    hop_length = _cfg["data"]["hop_length"]
    filter_length = _cfg["data"]["filter_length"]
    segment_size = _cfg["train"]["segment_size"]
    seg_frames = segment_size // hop_length

    # Why this filter exists:
    # RVC's data_utils computes len_min = min(2 * feat_frames, spec_frames),
    # then truncates wav to len_min * hop_length. During training,
    # rand_slice_segments picks an offset from [0, len_min - seg_frames + 1)
    # *in spec-frame units*, then slice_segments(wave, offset * hop, segment_size)
    # is called on the wav. spectrogram_torch with center=False produces
    # spec_frames = (n_samples - filter_length) // hop_length + 1, which is
    # smaller than n_samples // hop_length. If 2*feat_frames > spec_frames,
    # the chosen offset can land near the end of spec but past the end of the
    # already-truncated wav, and slice_segments returns a 0-length tensor.
    # Crashes with "Tensor sizes: [0]".
    #
    # Safe condition: len_min - seg_frames >= 0 AND
    # the worst-case wav slice end (== len_min * hop_length) fits in wav.
    # Since wav is truncated to exactly len_min * hop, the second part is
    # automatic. We just need len_min >= seg_frames + small margin.
    # 3x margin: empirically RVC still crashes with [0]-length slice when
    # len_min is just above seg_frames, even though the math says it shouldn't.
    # Likely involves an off-by-one in the dataloader's bucket sampler. 3x is
    # cheap on a 264-slice dataset and rules out the crash.
    min_len_min = seg_frames * 3
    min_wav_samples = (min_len_min - 1) * hop_length + filter_length

    print(
        f"Filelist filter: seg_frames={seg_frames}, required len_min>={min_len_min} "
        f"(min wav samples={min_wav_samples}, ~{min_wav_samples/sample_rate:.2f}s)"
    )

    import soundfile as sf

    filelist_path = log_dir / "filelist.txt"
    lines = []
    skipped_short_wav = 0
    skipped_short_feat = 0
    skipped_missing = 0
    for wav_file in sorted(list(gt_wavs_dir.glob("*.wav"))):
        name_stem = wav_file.name
        f0_file = f0_dir / f"{name_stem}.npy"
        f0nsf_file = f0nsf_dir / f"{name_stem}.npy"
        feat_file = feature_dir / f"{name_stem.replace('.wav', '.npy')}"

        if not (f0_file.exists() and f0nsf_file.exists() and feat_file.exists()):
            skipped_missing += 1
            continue

        # Compute the same spec_frames RVC's data_utils will compute.
        n_samples = sf.info(str(wav_file)).frames
        spec_frames = max(0, (n_samples - filter_length) // hop_length + 1)

        feat_frames = np.load(feat_file).shape[0]
        phone_frames = feat_frames * 2  # data_utils repeats by 2
        len_min = min(phone_frames, spec_frames)

        if len_min < min_len_min:
            if spec_frames < min_len_min:
                skipped_short_wav += 1
            else:
                skipped_short_feat += 1
            continue

        # Stale .spec.pt cache may carry an old length; remove so RVC recomputes.
        spec_cache = wav_file.with_suffix('.spec.pt')
        if spec_cache.exists():
            spec_cache.unlink()

        p_wav = os.path.relpath(wav_file, rvc_src)
        p_feat = os.path.relpath(feat_file, rvc_src)
        p_f0 = os.path.relpath(f0_file, rvc_src)
        p_f0nsf = os.path.relpath(f0nsf_file, rvc_src)
        lines.append(f"{p_wav}|{p_feat}|{p_f0}|{p_f0nsf}|0")

    print(
        f"Filelist: {len(lines)} kept, "
        f"{skipped_short_wav} skipped (wav too short), "
        f"{skipped_short_feat} skipped (features too short), "
        f"{skipped_missing} skipped (missing features)."
    )
    if not lines:
        print("Error: no usable training slices after filtering. Add more source audio.")
        sys.exit(1)

    with open(filelist_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # Step E: Prepare Training Config
    config_tmpl = rvc_src / "configs" / "v2" / f"{sample_rate//1000}k.json"
    if not config_tmpl.exists():
        config_tmpl = rvc_src / "configs" / "v2" / "48k.json"
    
    with open(config_tmpl, "r") as f:
        config = json.load(f)
    
    config["train"]["batch_size"] = batch_size
    config["data"]["sampling_rate"] = sample_rate
    config["data"]["training_files"] = os.path.relpath(filelist_path, rvc_src)
    
    # Increase logging frequency for CPU training
    config["train"]["log_interval"] = 10 
    
    with open(log_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Step F: Train Model
    pretrain_g = pretrained_dir / f"f0G{sample_rate//1000}k.pth"
    pretrain_d = pretrained_dir / f"f0D{sample_rate//1000}k.pth"

    cache_flag = "1" if (cache_in_gpu and "cuda" in device) else "0"
    run_command([
        python_exe, str(rvc_src / "infer/modules/train/train.py"),
        "-e", name,
        "-sr", f"{sample_rate//1000}k",
        "-f0", "1",
        "-bs", str(batch_size),
        "-g", "0" if "cuda" in device else "-",
        "-v", "v2",
        "-sw", "1",
        "-l", "1",
        "-te", str(epochs),
        "-se", str(save_every),
        "-pg", str(pretrain_g),
        "-pd", str(pretrain_d),
        "-c", cache_flag,
    ], cwd=rvc_src)

    # Step G: Index Build — run in a subprocess so the trainer's residual
    # torch/numpy allocations are released before faiss concatenates every
    # feature file into one big array.
    print("Building Faiss index...")
    run_command([
        python_exe,
        str(Path(__file__).parent / "utils" / "build_index.py"),
        str(feature_dir),
        str(trained_dir / f"{name}.index"),
    ])
    
    # Final weight copy
    weights_folder = rvc_src / "assets" / "weights"
    if weights_folder.exists():
        pth_file = weights_folder / f"{name}.pth"
        if pth_file.exists():
            shutil.copy(pth_file, trained_dir / f"{name}.pth")
            print(f"Final model copied to {trained_dir / f'{name}.pth'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--pretrained", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--name", type=str, default="canary")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--sample-rate", type=int, default=48000)
    parser.add_argument("--reprep", action="store_true")
    parser.add_argument("--save-every", type=int, default=50, dest="save_every",
                        help="checkpoint frequency in epochs")
    parser.add_argument("--top-db", type=int, default=30, dest="top_db",
                        help="silence threshold (dB) for slicing — higher = more aggressive trim")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--cache-in-gpu", action="store_true", dest="cache_in_gpu",
                        help="keep dataset in GPU memory (CUDA only, needs >12 GB VRAM)")
    args = parser.parse_args()

    train(
        Path(args.source),
        Path(args.dataset),
        Path(args.pretrained),
        Path(args.out),
        args.name,
        args.epochs,
        args.batch_size,
        args.sample_rate,
        args.reprep,
        save_every=args.save_every,
        top_db=args.top_db,
        device_pref=args.device,
        cache_in_gpu=args.cache_in_gpu,
    )
