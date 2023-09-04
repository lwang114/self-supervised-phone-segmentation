#!/usr/bin/env python3 -u

import argparse
import os
import os.path as osp
import tqdm
import torch

from npy_append_array import NpyAppendArray

import fairseq
import soundfile as sf


def get_parser():
    parser = argparse.ArgumentParser()  
    parser.add_argument('data', help='location of tsv files')
    parser.add_argument('--split', help='which split to read', required=True)
    parser.add_argument('--save-dir', help='where to save the output', required=True)
    parser.add_argument('--checkpoint', type=str, help='checkpoint for wav2vec ctc model', required=True)

    return parser


def FeatureReader(object):
    def __init__(self, cp_file):
        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task(
            [cp_file]
        )
        model = model[0]
        model.eval()
        model.cuda()
        self.model = model
        self.task = task

    def read_audio(self, fname):
        """Load an audio file and return PCM along with the sample rate"""
        wav, sr = sf.read(fname)
        assert sr == 16e3

        return wav
 
    def get_feats(self, loc):
        x = self.read_audio(loc)
        with torch.no_grad():
            source = torch.from_numpy(x).float().cuda()
            if self.task.cfg.normalize:
                assert source.dim() == 1, source.dim()
                with torch.no_grad():
                    source = F.layer_norm(source, source.shape)
            source = source.view(1, -1)

            results = self.model(source=source, mask=False, features_only=True)
            zeros = torch.zeros_like(results["x"])
            features = [r[0].squeeze(0).cpu() if r[0] is not None else zeros.clone() for r in results["layer_results"]]
            return features


def get_iterator(args):
    with open(osp.join(args.data, args.split) + ".tsv", "r") as fp:
        lines = fp.read().split("\n")
        root = lines.pop(0).strip()
        files = [osp.join(root, line.split("\t")[0]) for line in lines if len(line) > 0]

        num = len(files)
        reader = FeatureReader(args.checkpoint)

        def iterate():
            for fname in files:
                feats = reader.get_feats(fname)
                yield feats

    return iterate, num


def main():
    parser = get_parser()
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    def create_files(dest):
        if osp.exists(dest + ".npy"):
            os.remove(dest + ".npy")
        npaa = NpyAppendArray(dest + ".npy")
        return npaa

    save_paths = [
        osp.join(args.save_dir, f"feat_layer{i+1}", args.split)
        for i in range(12)
    ]
    npaa_list = [create_files(save_path) for save_path in save_paths]
    sizes_list = ['' for _ in range(12)]

    generator, num = get_iterator(args)
    iterator = generator()

    for feats in tqdm.tqdm(iterator, total=num):
        for i, feat in enumerate(feats):
            print(len(feat), file=l_f)
            
            if len(feat) > 0:
                npaa_list[i].append(feat.numpy())
                sizes_list[i] += str(len(feat))+"\n"

    for i, save_path in enumerate(save_paths):
        with open(save_path + ".lengths", "w") as l_f:
            l_f.write("".join(sizes_list[i]))

if __name__ == "__main__":
    main()

