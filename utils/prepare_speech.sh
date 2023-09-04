#!/bin/bash

wav_dir=$1
align_dir=$2
unsupseg_dir=$3
tgt_dir=$4
model=$5

stage=4
stop_stage=4

train_split="train"
all_splits="train valid test"
if [ ! -d $tgt_dir ]; then
    mkdir -p $tgt_dir
fi
 
FAIRSEQ_ROOT=/home/hertin/workplace/wav2vec/fairseq
echo "prepare_speech.sh stage 1: extract .tsv files"
if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
    for x in train test; do
        python utils/extract_timit_tsv.py --timit_path $wav_dir/$x --out_path $tgt_dir/$x.tsv
    done

    if [ ! -f $tgt_dir/valid.tsv ]; then
        cp $tgt_dir/test.tsv $tgt_dir/valid.tsv
    fi
fi

echo "prepare_speech.sh stage 2: extract _gt.src files"
if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
    for x in train test; do
        python utils/extract_timit_alignment.py \
            --manifest_dir $tgt_dir \
            --align_dir $align_dir/$x \
            --out_dir $tgt_dir/phn_gt_seg \
            --split $x
    done
    cp $tgt_dir/phn_gt_seg/test_gt.src $tgt_dir/phn_gt_seg/valid_gt.src
fi

echo "prepare_speech.sh stage 3: extract .src files"
if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
    seg_dir=$unsupseg_dir/runs/timit_unsup_seg
    out_dir=$tgt_dir/phn_unsup_seg
    if [ ! -d $out_dir ]; then
        mkdir -p $out_dir  
    fi 

    for x in train valid test; do
        tsv_path=$tgt_dir/$x.tsv
        python utils/prepare_timit_unsup_seg.py \
            --in-dir $seg_dir \
            --out-path $out_dir/$x.src \
            --tsv-path $tsv_path
    done
fi

echo "prepare_speech.sh stage 4: extract speech features"
if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then 
    for split in $all_splits; do
        python utils/extract_speech_features.py $tgt_dir \
            --split $split \
            --save-dir $tgt_dir \
            --checkpoint $model
    done 
fi
