import sys
from pathlib import Path
import numpy as np
import faiss


def build_index(feature_dir: Path, out_path: Path) -> None:
    feature_files = list(feature_dir.glob("*.npy"))
    if not feature_files:
        print("No feature files found; skipping index build.")
        return

    feats = np.concatenate([np.load(f) for f in feature_files], axis=0)
    n_features, dim = feats.shape
    nlist = max(1, n_features // 39)
    index = faiss.IndexIVFFlat(faiss.IndexFlatL2(dim), dim, nlist)
    index.train(feats)
    index.add(feats)
    faiss.write_index(index, str(out_path))
    print(f"Wrote index ({n_features} vectors, dim={dim}) to {out_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: build_index.py <feature_dir> <out_index_path>")
        sys.exit(2)
    build_index(Path(sys.argv[1]), Path(sys.argv[2]))
