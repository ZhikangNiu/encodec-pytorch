# encodec-pytorch
This is an unofficial implementation of the paper [High Fidelity Neural Audio Compression](https://arxiv.org/pdf/2210.13438.pdf) in PyTorch.

The LibriTTS960h 24khz encodec checkpoint is release in https://huggingface.co/zkniu/encodec-pytorch/tree/main

I hope we can get together to do something meaningful and rebuild encodec in this repo.

## Introduction
This repository is based on [encodec](https://github.com/facebookresearch/encodec) and [EnCodec_Trainer](https://github.com/Mikxox/EnCodec_Trainer).

Based on the [EnCodec_Trainer](https://github.com/Mikxox/EnCodec_Trainer), I have made the following changes:
- support multi-gpu training.
- support hydra configuration management.
- align the loss functions and hyperparameters.
- support warmup scheduler in training.
- support the test script to test the model.

TODO:
- [ ] support the 48khz model.
- [ ] support wandb or tensorboard to monitor the training process.

## Enviroments
The code is tested on the following environment:
- Python 3.9
- PyTorch 2.0.0 (You can try other versions, but I can't guarantee that it will work. Because torch have changed some api default value (eg: stft). )
- GeForce RTX 3090 x 4

In order to you can run the code, you can install the environment by the help of requirements.txt.

## Usage
### Training
#### 1. Prepare dataset
I use the librispeech as the train datasets and use the `datasets/generate_train_file.py` generate train csv which is used in the training process. You can check the `datasets/generate_train_file.py` and `customAudioDataset.py` to understand how to prepare your own dataset.
Also you can use `ln -s` to link the dataset to the `datasets` folder.

#### 2. Train
You can use the following command to train the model using multi gpu:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_multi_gpu.py \
                        distributed.torch_distributed_debug=False \
                        distributed.find_unused_parameters=True \
                        distributed.world_size=4 \
                        common.save_interval=5 \
                        common.max_epoch=100 \
                        datasets.tensor_cut=100000 \
                        datasets.batch_size=8 \
                        datasets.train_csv_path=./datasets/librispeech_train100h.csv \
                        lr_scheduler.warmup_epoch=20 \
                        optimization.lr=5e-5 \
                        optimization.disc_lr=5e-5 \
```
Note: 
1. if you set a small `datasets.tensor_cut`, you can set a large `datasets.batch_size` to speed up the training process.
2. if you encounter bug about `RuntimeError(f"Mismatch in number of params: ours is {len(params)}, at least one worker has a different one.")`. You can use a small `datasets.tensor_cut` to solve this problem.
3. if your torch version is lower 1.8, you need to check the default value of `torch.stft(return_complex)` in the `audio_to_mel.py`  
4. if you encounter bug about multi-gpu training, you can try to set `distributed.torch_distributed_debug=True` to get more message about this problem.
5. the single gpu training method is similar to the multi-gpu training method, you only need to set the `distributed.data_parallel=False` parameter to the command, like this:
    ```bash
    python train_multi_gpu.py distributed.data_parallel=False
                        common.save_interval=5 \
                        common.max_epoch=100 \
                        datasets.tensor_cut=1000 \
                        datasets.batch_size=2 \
                        datasets.train_csv_path=YOUR_PATH/train_encodec/datasets/librispeech_train100h.csv \
                        lr_scheduler.warmup_epoch=20 \
                        optimization.lr=5e-5 \
                        optimization.disc_lr=5e-5 \
    ```
6. the loss is not converged to zero, but the model can be used to compress and decompress the audio. you can use the `compression.sh` to test your model in every log_interval epoch.
7. the original paper dataset is larger than 17000h, but I only use LibriTTS960h to train the model, so the model is not good enough. If you want to train a better model, you can use the larger dataset.
8. **The code is not well tested, so there may be some bugs. If you encounter any problems, you can open an issue or contact me by email.**
### Test
I have add a shell script to compress and decompress the audio by different bandwidth, you can use the `compression.sh` to test your model. 

The script can be used as follows:
```shell
sh compression.sh INPUT_WAV_FILE [MODEL_NAME] [CHECKPOINT]
```
- INPUT_WAV_FILE is the wav file you want to test
- MODEL_NAME is the model name, default is `encodec_24khz`,support `encodec_48khz`, `my_encodec`,`encodec_bw`
- CHECKPOINT is the checkpoint path, when your MODEL_NAME is `my_encodec`,you can point out the checkpoint

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

