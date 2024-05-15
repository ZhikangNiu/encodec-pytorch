#!/bin/bash
#SBATCH -p hocuspocus
#SBATCH --qos=enthusiast
#SBATCH --mem=0
#SBATCH --time=0
#SBATCH --cpus-per-task=72
#SBATCH --gpus=1
#SBATCH -J encodec_train

srun --container-name=encodec_arm_2 --container-mounts=/hpi/fs00/share/fg-meinel/datasets/audio:/home/audio,/hpi/fs00/home/leonardo.auri/:/home/leonardo.auri/ --container-workdir=/home/leonardo.auri/encodec-pytorch/ \
        --container-writable \
        python train_multi_gpu.py \
        distributed.data_parallel=False \
        common.save_interval=1 \
        common.test_interval=1 \
        common.max_epoch=150 \
        common.log_interval=1000 \
        datasets.tensor_cut=96_000 \
        datasets.batch_size=32 \
        datasets.num_workers=64 \
        datasets.train_csv_path=/home/leonardo.auri/encodec-pytorch/datasets/all_ds_train.csv \
        datasets.test_csv_path=/home/leonardo.auri/encodec-pytorch/datasets/all_ds_test.csv \
        lr_scheduler.warmup_epoch=1 \
        model.sample_rate=48_000 \
        model.target_bandwidths="[3., 6., 12., 24.]" \
        model.causal=False \
        model.norm=time_group_norm \
        model.segment=1. \
        model.name=encodec_48khz_reproduce \
        model.channels=2