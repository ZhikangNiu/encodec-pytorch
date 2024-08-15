CODE_ROOT="$PWD"
ENCODEC_SET=encodec320x_ratios8542
BATCH_SIZE=6
MAX_EPOCH=150
TENSOR_CUT=72000
WARMUP_EPOCH=1
DATASET=lt960
LR=3e-4
python train_multi_gpu.py distributed.data_parallel=False \
                    common.save_interval=1 \
                    common.test_interval=1 \
                    common.max_epoch=${MAX_EPOCH} \
                    common.log_interval=1 \
                    datasets.tensor_cut=${TENSOR_CUT} \
                    model.ratios=[8,5,4,2] \
                    datasets.batch_size=${BATCH_SIZE} \
                    datasets.train_csv_path=${CODE_ROOT}/datasets/libritts960_train_all.csv \
                    datasets.test_csv_path=${CODE_ROOT}/datasets/libritts_test_all.csv \
                    datasets.fixed_length=500 \
                    datasets.num_workers=8 \
                    lr_scheduler.warmup_epoch=${WARMUP_EPOCH} \
                    optimization.lr=${LR} \
                    optimization.disc_lr=${LR} \
                    checkpoint.save_folder=/home/v-zhikangniu/encodec-pytorch/outputs/${ENCODEC_SET}_${DATASET}_bs${BATCH_SIZE}_tc$((${TENSOR_CUT} / 1000))_lr${LR}_wup${WARMUP_EPOCH} \
                    hydra.run.dir=/home/v-zhikangniu/encodec-pytorch/outputs/${ENCODEC_SET}_${DATASET}_bs${BATCH_SIZE}_tc$((${TENSOR_CUT} / 1000))_lr${LR}_wup${WARMUP_EPOCH} \
                    model.disc_n_ffts=[2048,1024,512,256,128] \
                    model.disc_win_lengths=[2048,1024,512,256,128] \
                    model.disc_hop_lengths=[512,256,128,64,32]