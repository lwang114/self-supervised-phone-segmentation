from copy import deepcopy
import torch
from torch.utils.data import Dataset
from os.path import join
from boltons.fileutils import iter_find_files
import torchaudio
import math
from functools import partial
from torch.utils.data import DataLoader
import numpy as np
import random

#LAYERS = [f'feat_layer{i}/precompute_pca512' for i in range(14, 15)]
LAYERS = [f'feat_layer{i}/precompute_pca512' for i in range(1, 25)]
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def collate_fn_padd(batch):
    wavs = [t[0] for t in batch]
    sr = [t[1] for t in batch]
    seg_raw = [t[2] for t in batch]
    seg_aligned = [t[3] for t in batch]
    phonemes = [t[4] for t in batch]
    bin_labels = [t[5] for t in batch]
    lengths = [t[6] for t in batch]
    fnames = [t[7] for t in batch]
    padded_wavs = torch.nn.utils.rnn.pad_sequence(wavs, batch_first=True, padding_value=0)
    padded_bin_labels = torch.nn.utils.rnn.pad_sequence(bin_labels, batch_first=True, padding_value=0)
    return padded_wavs, seg_aligned, padded_bin_labels, phonemes, lengths, fnames

def spectral_size(wav_len, layers):
    for kernel, stride, padding in layers:
        wav_len = math.floor((wav_len + 2*padding - 1*(kernel-1) - 1)/stride + 1)
    return wav_len

def construct_mask(lengths, device):
    lengths = torch.tensor(lengths)
    max_len = lengths.max()
    mask = torch.arange(max_len).expand(len(lengths), max_len) < lengths.unsqueeze(1)
    return mask.to(device)

def get_subset(dataset, percent):
    A_split = int(len(dataset) * percent)
    B_split = len(dataset) - A_split
    dataset, _ = torch.utils.data.random_split(dataset, [A_split, B_split])
    return dataset

def get_dloaders(cfg, logger, layers=LAYERS, g=None, is_training=True):
    if cfg.data == "timit":
        """
        train, val, test = TrainTestDataset.get_datasets(
            path=cfg.timit_path, 
            val_ratio=cfg.val_ratio, 
            train_percent=cfg.train_percent, 
            layers=layers, 
        )
        """
        train, val, test = TrainValTestDataset.get_datasets(
            path=cfg.timit_path, 
            layers=layers, 
            train_percent=cfg.train_percent,
        )
    elif cfg.data == "buckeye":
        train, val, test = TrainValTestDataset.get_datasets(
            path=cfg.buckeye_path,
            layers=layers, 
            train_percent=cfg.train_percent,
        )
    else:
        raise ValueError("Provided dataset not supported")

    logger.info("Train set size: {}".format(len(train)))
    logger.info("Val set size: {}".format(len(val)))
    logger.info("Test set size: {}".format(len(test)))
    
    trainloader = DataLoader(
        train, 
        batch_size=cfg.batch_size, 
        shuffle=is_training, 
        num_workers=cfg.num_workers,
        worker_init_fn=seed_worker,
        generator=g,
        collate_fn=collate_fn_padd
    )
    valloader = DataLoader(
        val, 
        batch_size=cfg.batch_size, 
        shuffle=False, 
        num_workers=cfg.num_workers,
        worker_init_fn=seed_worker,
        generator=g,
        collate_fn=collate_fn_padd
    )
    testloader = DataLoader(
        test, 
        batch_size=cfg.batch_size, 
        shuffle=False, 
        num_workers=cfg.num_workers,
        worker_init_fn=seed_worker,
        generator=g,
        collate_fn=collate_fn_padd
    )
    return trainloader, valloader, testloader

class ExtractedPhnDataset(Dataset):
    def __init__(self, paths):
        self.paths = paths
        self.data = []
        for path in paths:
            self.data.append(np.load(path+".npy"))
            with open(path+".lengths", "r") as f_len:
                sizes = f_len.read().strip().split("\n")
                self.sizes = list(map(int, sizes))
                self.offsets = []
                offset = 0
                for size in self.sizes:
                    self.offsets.append(offset)
                    offset += size

            self.phonemes = []
            self.times = []
            self.scaled_times = []
            self.bin_labels = []
            with open(path+"_gt.src", "r") as gt_f:
                for line in gt_f:
                    clusts = list(map(int, line.rstrip().split()))
                    clusts = torch.tensor(clusts)
                    clusts, _, counts = clusts.unique_consecutive(return_inverse=True, return_counts=True)
                    phonemes = list(map(lambda x:str(x.item()), clusts))
                    times = []
                    scaled_times = []
                    offset = 0
                    for c in counts:
                        times.append((offset*320, (offset+c.item())*320))
                        scaled_times.append((offset, offset+c.item()))
                        offset += c.item()
                    self.phonemes.append(phonemes)
                    self.times.append(times)
                    self.scaled_times.append(scaled_times)

            with open(path+".src", "r") as src_f:
                for idx, line in enumerate(src_f):
                    clusts = list(map(int, line.rstrip().split()))
                    clusts = torch.tensor(clusts)
                    _, _, counts = clusts.unique_consecutive(return_inverse=True, return_counts=True)
                    pred_segments = []
                    offset = counts[0].item()
                    for c in counts[1:]:
                        pred_segments.append(offset)
                        offset += c.item()
 
                    size = max(self.sizes[idx], len(clusts))
                    bin_labels = torch.zeros(size).float()
                    bin_labels[pred_segments] = 1.0
                    if self.sizes[idx] < len(bin_labels):
                        bin_labels = bin_labels[:self.sizes[idx]]
                    
                    self.bin_labels.append(bin_labels)
        super(ExtractedPhnDataset, self).__init__()

    @staticmethod
    def get_datasets(path):
        raise NotImplementedError

    def process_file(self, idx):
        offset = self.offsets[idx]
        size = self.sizes[idx]
        audio = [
            torch.tensor(d[offset:offset+size])
            for d in self.data
        ]
        audio = torch.stack(audio, dim=1)
        
        phonemes = self.phonemes[idx]
        times = torch.FloatTensor(self.times[idx])
        scaled_times = torch.FloatTensor(self.scaled_times[idx])
        bin_labels = torch.FloatTensor(self.bin_labels[idx])
        return audio, 16e3, times, scaled_times, bin_labels, phonemes, str(idx)

    def __getitem__(self, idx):
        audio, sr, seg, seg_scaled, bin_labels, phonemes, fname = self.process_file(idx)
        return audio, sr, seg, seg_scaled, phonemes, bin_labels, len(audio), fname

    def __len__(self):
        return len(self.sizes)


class TrainTestDataset(ExtractedPhnDataset):
    def __init__(self, paths, layers=LAYERS, files=None):
        super(TrainTestDataset, self).__init__(paths)

    @staticmethod
    def get_datasets(path, val_ratio=0.1, train_percent=1.0, layers=LAYERS, files=None):
        train_dataset = TrainTestDataset([join(path, l, 'train') for l in layers])
        test_dataset  = TrainTestDataset([join(path, l, 'valid') for l in layers])
        train_len   = len(train_dataset)
        train_split = int(train_len * (1 - val_ratio))
        val_split   = train_len - train_split
        train_holdout = int(train_split * (1 - train_percent))
        train_split -= train_holdout
        dataset_copy = deepcopy(train_dataset)
        train_dataset, val_dataset, _ = torch.utils.data.random_split(train_dataset, [train_split, val_split, train_holdout])
        train_dataset.dataset = dataset_copy
        train_dataset.path = join(path, 'train')
        val_dataset.path = join(path, 'train')
        return train_dataset, val_dataset, test_dataset


class TrainValTestDataset(ExtractedPhnDataset):
    def __init__(self, paths, layers=LAYERS, files=None):
        super(TrainValTestDataset, self).__init__(paths)

    @staticmethod
    def get_datasets(path, layers=LAYERS, files=None, train_percent=1.0):
        train_dataset = TrainValTestDataset([join(path, l, 'train') for l in layers])
        if train_percent != 1.0:
            train_dataset = get_subset(train_dataset, train_percent)
            train_dataset.path = join(path, 'train')
        val_dataset = TrainValTestDataset([join(path, l, 'valid') for l in layers])
        test_dataset = TrainValTestDataset([join(path, l, 'test') for l in layers])

        return train_dataset, val_dataset, test_dataset
