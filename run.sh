#!/bin/bash
#SBATCH --job-name="logs/timit_phone_segment"
#SBATCH --output="logs/%j.%N_timit_phone_segment.out"
#SBATCH --error="logs/%j.%N_timit_phone_segment.err"
#SBATCH --partition=gpu
#SBATCH --mem-per-cpu=2400
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --sockets-per-node=1
#SBATCH --cores-per-socket=4
#SBATCH --threads-per-core=4
#SBATCH --export=ALL
#SBATCH --gres=gpu:v100:1

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate /home/hertin/.conda/envs/wav2vec

stage=3
stop_stage=3

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
    for i in {1..24} 
    do
        for x in train valid; do
            cp ../manifest/timit_unsup_seg/matched/phn_gt_seg/$x.src ../manifest/timit_unsup_seg/matched/feat_layer$i/${x}_gt.src
            cp ../manifest/timit_unsup_seg/matched/phn_unsup_seg/$x.src ../manifest/timit_unsup_seg/matched/feat_layer$i/${x}.src
            cp ../manifest/timit_unsup_seg/matched/phn_gt_seg/$x.src ../manifest/timit_unsup_seg/matched/feat_layer$i/precompute_pca512/${x}_gt.src
            cp ../manifest/timit_unsup_seg/matched/phn_unsup_seg/$x.src ../manifest/timit_unsup_seg/matched/feat_layer$i/precompute_pca512/${x}.src
        done
    done
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
    python run_extracted.py
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
    python run.py
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
    python predict.py
fi
