# encodec-pytorch
>[!IMPORTANT]
>This is an unofficial implementation of the paper [High Fidelity Neural Audio Compression](https://arxiv.org/pdf/2210.13438.pdf) in PyTorch.
>
>The LibriTTS960h 24khz encodec checkpoint and disc checkpoint is release in https://huggingface.co/zkniu/encodec-pytorch/tree/main
>
>I hope we can get together to do something meaningful and rebuild encodec in this repo.

## Introduction
This repository is based on [encodec](https://github.com/facebookresearch/encodec) and [EnCodec_Trainer](https://github.com/Mikxox/EnCodec_Trainer).

Based on the [EnCodec_Trainer](https://github.com/Mikxox/EnCodec_Trainer), I have made the following changes:
- support multi-gpu training.
- support AMP training (you need to reduce learning rate and scale vq epsilon from 1e-5 to 1e-3, the reason you can check [issue 8](https://github.com/ZhikangNiu/encodec-pytorch/issues/8))
  - Couldn't work, so don't use amp
- support hydra configuration management.
- align the loss functions and hyperparameters.
- support warmup scheduler in training.
- support the test script to test the model.
- support tensorboard to monitor the training process.
- support 48khz and stereo models, thanks [@leoauri](https://github.com/leoauri) in https://github.com/ZhikangNiu/encodec-pytorch/pull/22.
- support slurm training, thanks [@leoauri](https://github.com/leoauri). in https://github.com/ZhikangNiu/encodec-pytorch/pull/22.
- support loss balancer, thanks [@leoauri](https://github.com/leoauri). in https://github.com/ZhikangNiu/encodec-pytorch/pull/22.
- You can find all the training scripts in scripts folder

## Enviroments
The code is tested on the following environment:
- Python 3.9
- PyTorch 2.0.0 / PyTorch 1.13
- GeForce RTX 3090 x 4 / V100-16G x 8 / A40 x 3 / A100 x 1

In order to you can run the code, you can install the environment by the help of requirements.txt.

## Usage
### Training
#### 1. Prepare dataset
I use the librispeech as the train datasets and use the `datasets/generate_train_file.py` generate train csv which is used in the training process. You can check the `datasets/generate_train_file.py` and `customAudioDataset.py` to understand how to prepare your own dataset.
Also you can use `ln -s` to link the dataset to the `datasets` folder.
#### [Optional] Docker image
I provide a dockerfile to build a docker image with all the necessary dependencies.
1. Building the image
```shell
docker build -t encodec:v1 .
```
2. Using the image
```shell
# CPU running
docker run encodec:v1 <command> # you can add some parameters, such as -tid
# GPU running
docker run --gpus=all encodec:v1 <command>
```
#### 2. Train
You can use the following command to train the model using multi gpu:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_multi_gpu.py \
                        distributed.torch_distributed_debug=False \
                        distributed.find_unused_parameters=True \
                        distributed.world_size=4 \
                        common.save_interval=2 \
                        common.test_interval=2 \
                        common.max_epoch=100 \
                        datasets.tensor_cut=100000 \
                        datasets.batch_size=8 \
                        datasets.train_csv_path=YOUR TRAIN DATA.csv \
                        lr_scheduler.warmup_epoch=20 \
                        optimization.lr=5e-5 \
                        optimization.disc_lr=5e-5 \
```
Note: 
1. if you set a small `datasets.tensor_cut`, you can set a large `datasets.batch_size` to speed up the training process.
2. when you are training on your own dataset, I suggest you need to choose a moderate-length audio, because If you train your encodec with 1 senconds tensorcut in a small dataset and the encodec model dosen't perform well.
2. if you encounter bug about `RuntimeError(f"Mismatch in number of params: ours is {len(params)}, at least one worker has a different one.")`. You can use a small `datasets.tensor_cut` to solve this problem.
3. if your torch version is lower 1.8, you need to check the default value of `torch.stft(return_complex)` in the `audio_to_mel.py`  
4. if you encounter bug about multi-gpu training, you can try to set `distributed.torch_distributed_debug=True` to get more message about this problem.
5. the single gpu training method is similar to the multi-gpu training method, you only need to set the `distributed.data_parallel=False` parameter to the command, like this:
    ```bash
        python train_multi_gpu.py distributed.data_parallel=False
                            common.save_interval=5 \
                            common.max_epoch=100 \
                            datasets.tensor_cut=72000 \
                            datasets.batch_size=4 \
                            datasets.train_csv_path=YOUR TRAIN DATA.csv \
                            lr_scheduler.warmup_epoch=10 \
                            optimization.lr=5e-5 \
                            optimization.disc_lr=5e-5 \
    ```
6. the loss is not converged to zero, but the model can be used to compress and decompress the audio. you can use the `compression.sh` to test your model in every log_interval epoch.
7. the original paper dataset is larger than 17000h, but I only use LibriTTS960h to train the model, so the model is not good enough. If you want to train a better model, you can use the larger dataset.
8. **The code is not well tested, so there may be some bugs. If you encounter any problems, you can open an issue or contact me by email.**
9. When I add AMP training, I found the RVQ loss always be `nan`, and I use L2 norm to normalized quantize and x, like the code -> actually, it's unstable.
    ```python
        quantize = F.normalize(quantize)  
        commit_loss = F.mse_loss(quantize.detach(), x)
    ``` 
11. When you try to use amp training, you need to reduce learning rate and scale vq epsilon from 1e-5 to 1e-3, the reason you can check [issue 8](https://github.com/ZhikangNiu/encodec-pytorch/issues/8)
12. I suggest you need to focus on the generator loss, the commit loss it could be not converge, you can check some objective metrics about pesq, stoi.

#### Slurm
Usage will depend on your cluster setup, but see `scripts/train.sbatch` for an example. This uses a container with the dependencies installed. Run `sbatch scripts/train.sbatch` from the repository root to use.

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

## Acknowledgement
Thanks to the following repositories:
- [encodec](https://github.com/facebookresearch/encodec)
- [EnCodec_Trainer](https://github.com/Mikxox/EnCodec_Trainer)
- [melgan-neurips](https://github.com/descriptinc/melgan-neurips): audio_to_mel.py

## LICENSE
The code is same as [encodec](https://github.com/facebookresearch/encodec) LICENSE.

