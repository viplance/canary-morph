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

def train(source_dir: Path, dataset_dir: Path, pretrained_dir: Path, trained_dir: Path, name: str, epochs: int, batch_size: int, sample_rate: int, reprep: bool):
    rvc_src = trained_dir.parent / "rvc-src"
    if not rvc_src.exists():
        print(f"Error: RVC source not found at {rvc_src}. Run setup first.")
        sys.exit(1)

    log_dir = rvc_src / "logs" / name
    log_dir.mkdir(parents=True, exist_ok=True)

    # 1. Pre-process dataset if needed
    if reprep or not (dataset_dir.exists() and any(dataset_dir.iterdir())):
        print(f"Preparing dataset slices from {source_dir}...")
        count = prepare_dataset(source_dir, dataset_dir, sample_rate)
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
    device = "cpu"
    if "cuda" in os.environ.get("DEVICE", "").lower():
        device = "cuda:0"
    elif "mps" in os.environ.get("DEVICE", "").lower():
        device = "mps"
    
    run_command([
        python_exe, str(rvc_src / "infer/modules/train/extract_feature_print.py"),
        device, "1", "0", str(log_dir), "v2", "False"
    ], cwd=rvc_src)

    # Step D: Generate filelist.txt
    gt_wavs_dir = log_dir / "0_gt_wavs"
    f0_dir = log_dir / "2a_f0"
    f0nsf_dir = log_dir / "2b-f0nsf"
    feature_dir = log_dir / "3_feature768"
    
    filelist_path = log_dir / "filelist.txt"
    lines = []
    for wav_file in sorted(list(gt_wavs_dir.glob("*.wav"))):
        name_stem = wav_file.name
        f0_file = f0_dir / f"{name_stem}.npy"
        f0nsf_file = f0nsf_dir / f"{name_stem}.npy"
        feat_file = feature_dir / f"{name_stem.replace('.wav', '.npy')}"
        
        if f0_file.exists() and f0nsf_file.exists() and feat_file.exists():
            p_wav = os.path.relpath(wav_file, rvc_src)
            p_feat = os.path.relpath(feat_file, rvc_src)
            p_f0 = os.path.relpath(f0_file, rvc_src)
            p_f0nsf = os.path.relpath(f0nsf_file, rvc_src)
            lines.append(f"{p_wav}|{p_feat}|{p_f0}|{p_f0nsf}|0")
    
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
        "-se", "50", 
        "-pg", str(pretrain_g), 
        "-pd", str(pretrain_d),
        "-c", "0"
    ], cwd=rvc_src)

    # Step G: Index Build
    print("Building Faiss index...")
    import faiss
    feature_files = list(feature_dir.glob("*.npy"))
    if feature_files:
        feats = [np.load(f) for f in feature_files]
        feats = np.concatenate(feats, axis=0)
        n_features = feats.shape[0]
        nlist = max(1, n_features // 39)
        index = faiss.IndexIVFFlat(faiss.IndexFlatL2(feats.shape[1]), feats.shape[1], nlist)
        index.train(feats)
        index.add(feats)
        faiss.write_index(index, str(trained_dir / f"{name}.index"))
    
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
    args = parser.parse_args()
    
    train(Path(args.source), Path(args.dataset), Path(args.pretrained), Path(args.out), args.name, args.epochs, args.batch_size, args.sample_rate, args.reprep)
