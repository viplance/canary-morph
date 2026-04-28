import os
import sys
import argparse
from pathlib import Path
import soundfile as sf
import torch
import numpy as np

def infer(input_path: Path, output_path: Path, model_pth: Path, index_path: Path, pitch: int, method: str):
    # Resolve absolute paths before changing directory
    input_path = input_path.resolve()
    output_path = output_path.resolve()
    model_pth = model_pth.resolve()
    index_path = index_path.resolve()

    # We need to add rvc-src to path to import its modules
    rvc_src = (Path(__file__).parent.parent / "models" / "rvc-src").resolve()
    if not rvc_src.exists():
        print(f"Error: RVC source not found at {rvc_src}")
        sys.exit(1)
    
    # CRITICAL: Fix RVC backend bugs and PyTorch MPS limitations
    os.environ["weight_root"] = ""
    os.environ["index_root"] = str(model_pth.parent)
    os.environ["rmvpe_root"] = str(rvc_src / "assets" / "rmvpe")
    os.environ["hubert_path"] = str(rvc_src / "assets" / "hubert" / "hubert_base.pt")
    
    # Enable CPU fallback for MPS (required for STFT on Mac)
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    
    # CRITICAL: RVC modules parse sys.argv on import. 
    # We must clear it to avoid "unrecognized arguments" errors.
    old_argv = sys.argv
    sys.argv = [old_argv[0]] 

    try:
        # Change working directory to rvc-src so internal relative paths work
        os.chdir(str(rvc_src))
        sys.path.append(str(rvc_src))
        
        from infer.modules.vc.modules import VC
        from configs.config import Config
        
        config = Config()
        config.device = "cpu"
        if torch.backends.mps.is_available():
            config.device = "mps"
        if torch.cuda.is_available():
            config.device = "cuda:0"
        
        config.is_half = True if "cuda" in config.device else False
        
        vc = VC(config)
        vc.get_vc(str(model_pth))
        
        # opt_vocal is a tuple (info, (tgt_sr, audio_data))
        info, result = vc.vc_single(
            sid=0,
            input_audio_path=str(input_path),
            f0_up_key=pitch,
            f0_file=None,
            f0_method=method,
            file_index=str(index_path),
            file_index2="", 
            index_rate=0.75,
            filter_radius=3,
            resample_sr=0,
            rms_mix_rate=0.25,
            protect=0.33
        )
        
        if result[0] is None:
            print(f"Conversion failed: {info}")
            sys.exit(1)
            
        tgt_sr, audio_data = result
        sf.write(str(output_path), audio_data, tgt_sr, subtype='FLOAT')
        print(f"Inference complete. Saved to {output_path}")

    finally:
        # Restore argv just in case
        sys.argv = old_argv

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--index", type=str, required=True)
    parser.add_argument("--pitch", type=int, default=0)
    parser.add_argument("--method", type=str, default="rmvpe")
    args = parser.parse_args()
    
    infer(Path(args.input), Path(args.output), Path(args.model), Path(args.index), args.pitch, args.method)
