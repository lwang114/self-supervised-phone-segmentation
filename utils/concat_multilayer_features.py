import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dirs")
    parser.add_argument("--out_dir")
    parser.add_argument("--split")
    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()

    for in_dir in args.in_dirs.split()
    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    feat_path = in_dir / f"{args.split}.npy"
    feats = np.load(feat_path)
    offset = 0
    with open(in_dir / f"{args.split}.tsv", "r") as f_tsv,\
        open(in_dir / f"{args.split}.lengths", "r") as f_len:
        file_paths = f_tsv.read().strip().split("\n")
        _ = file_paths.pop(0)

        sizes = f_len.read().strip().split("\n")
        sizes = list(map(int, sizes))
        
        for fpath, size in tqdm(zip(file_paths, sizes)):
            prefix = Path(fpath).stem
            feat_path = out_dir / f"{prefix}.npy" 
            feat = feats[offset:offset+size]
            offset += size
            if not feat_path.exists():
                np.save(feat_path, feat)

if __name__ == "__main__":
    main()
