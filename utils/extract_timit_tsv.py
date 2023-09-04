import argparse
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timit_path")
    parser.add_argument("--out_path")
    args = parser.parse_args()

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f_tsv:
        print("./", file=f_tsv)
        for root, dirs, files in os.walk(args.timit_path):
            for fn in files:
                if fn.endswith(".wav"):
                    fpath = os.path.join(root, fn) 
                    print(fpath, file=f_tsv)

if __name__ == "__main__":
    main()
