# encodec-pytorch
This is an unofficial implementation of the paper [High Fidelity Neural Audio Compression](https://arxiv.org/pdf/2210.13438.pdf) in PyTorch.

## Introduction
This repository is based on [encodec](https://github.com/facebookresearch/encodec) and [EnCodec_Trainer](https://github.com/Mikxox/EnCodec_Trainer).

Based on the [EnCodec_Trainer](https://github.com/Mikxox/EnCodec_Trainer), I have made the following changes:
- support multi-gpu training.
- support hydra configuration management.
- align the loss functions and hyperparameters.
- support warmup scheduler in training.
- support the test script to test the model.

## Enviroments
The code is tested on the following environment:
- Python 3.9
- PyTorch 2.0.0 (You can try other versions, but I can't guarantee that it will work. Because torch have changed some api default value (eg: stft). )
- GeForce RTX 3090 x 4

In order to you can run the code, you can install the environment by the help of requirements.txt. You need to note that this environment contains some irrelevant packages, such as `s3prl`, `fairseq`, `torchvision`, etc. You should check the code and remove the irrelevant packages.

## Usage
### Training
#### 1. Prepare dataset
I use the librispeech as the train datasets and use the `datasets/generate_train_file.py` generate train csv which is used in the training process. You can check the `datasets/generate_train_file.py` and `customAudioDataset.py` to understand how to prepare your own dataset.

#### 2. Train
You can use the following command to train the model using multi gpu:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_multi_gpu.py \
                        distributed.torch_distributed_debug=False \
                        distributed.find_unused_parameters=True \
                        distributed.world_size=4 \
                        common.max_epoch=200 \
                        common.log_interval=5 \
                        common.max_epoch=100 \
                        datasets.tensor_cut=200000 \
                        datasets.train_csv_path=YOUR_PATH/librispeech_train100h.csv \
                        lr_scheduler.warmup_epoch=20 \
                        optimization.lr=1e-4 \
                        optimization.disc_lr=1e-4 \
```
### Test
I have add a shell script to compress and decompress the audio by different bandwidth, you can use the `compression.sh` to test your model. 

The script can be used as follows:
```shell
sh compression.sh INPUT_WAV_FILE [MODEL_NAME] [CHECKPOINT]
```
- INPUT_WAV_FILE is the wav file you want to test
- MODEL_NAME is the model name, default is `encodec_24khz`,support `encodec_48khz`, `my_encodec`
- CHECKPOINT is the checkpoint path, when your MODEL_NAME is `my_encodec`,you can point the checkpoint

if you want to test the model at a specific bandwidth, you can use the following command:
```shell
python main.py -r -b [bandwidth] -f [INPUT_FILE] [OUTPUT_WAV_FILE] -m [MODEL_NAME] -c [CHECKPOINT]
```
main.py from the [encodec](https://github.com/facebookresearch/encodec) , you can use the `-h` to check the help information.

#### 4. Acknowledgement
Thanks to the following repositories:
- [encodec](https://github.com/facebookresearch/encodec)
- [EnCodec_Trainer](https://github.com/Mikxox/EnCodec_Trainer)
- [melgan-neurips](https://github.com/descriptinc/melgan-neurips): audio_to_mel.py

