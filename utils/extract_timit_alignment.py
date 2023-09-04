import argparse
import os
from pathlib import Path

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest_dir")
    parser.add_argument("--align_dir") 
    parser.add_argument("--out_dir") 
    parser.add_argument("--split")
    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()
    manifest_dir = Path(args.manifest_dir)
    align_dir = Path(args.align_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frame_size = 320

    align_dict = dict()
    for root, dirs, files in os.walk(align_dir):
        for fn in files:
            if fn.lower().endswith(".phn"):
                fpath = Path(root) / Path(fn)
                uid = str(fpath).split(".phn")[0]
                units = []
                with fpath.open("r") as fp:
                    for i, line in enumerate(fp):
                        start, end, label = line.split()
                        start = int(start)
                        end = int(end)
                        s = int(start / frame_size)
                        e = int(end / frame_size)
                        units.extend([str(i)]*(e-s))
                align_dict[uid] = units

    with open(manifest_dir / f"{args.split}.tsv", "r") as f_tsv,\
        open(out_dir / f"{args.split}_gt.src", "w") as f_src:
        lines = f_tsv.read().strip().split("\n")
        _ = lines.pop(0)
        for l in lines:
            uid = l.split("\t")[0].split(".wav")[0]
            units = align_dict[uid]
            print(" ".join(units), file=f_src)
 
if __name__ == "__main__":
    main()
