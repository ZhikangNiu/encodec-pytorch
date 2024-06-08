# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Various utilities."""

import random
import typing as tp
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.multiprocessing as mp
import torchaudio


def _linear_overlap_add(frames: tp.List[torch.Tensor], stride: int):
    # Generic overlap add, with linear fade-in/fade-out, supporting complex scenario
    # e.g., more than 2 frames per position.
    # The core idea is to use a weight function that is a triangle,
    # with a maximum value at the middle of the segment.
    # We use this weighting when summing the frames, and divide by the sum of weights
    # for each positions at the end. Thus:
    #   - if a frame is the only one to cover a position, the weighting is a no-op.
    #   - if 2 frames cover a position:
    #          ...  ...
    #         /   \/   \
    #        /    /\    \
    #            S  T       , i.e. S offset of second frame starts, T end of first frame.
    # Then the weight function for each one is: (t - S), (T - t), with `t` a given offset.
    # After the final normalization, the weight of the second frame at position `t` is
    # (t - S) / (t - S + (T - t)) = (t - S) / (T - S), which is exactly what we want.
    #
    #   - if more than 2 frames overlap at a given point, we hope that by induction
    #      something sensible happens.
    assert len(frames)
    device = frames[0].device
    dtype = frames[0].dtype
    shape = frames[0].shape[:-1]
    total_size = stride * (len(frames) - 1) + frames[-1].shape[-1]

    frame_length = frames[0].shape[-1]
    t = torch.linspace(0, 1, frame_length + 2, device=device, dtype=dtype)[1: -1]
    weight = 0.5 - (t - 0.5).abs()

    sum_weight = torch.zeros(total_size, device=device, dtype=dtype)
    out = torch.zeros(*shape, total_size, device=device, dtype=dtype)
    offset: int = 0

    for frame in frames:
        frame_length = frame.shape[-1]
        out[..., offset:offset + frame_length] += weight[:frame_length] * frame
        sum_weight[offset:offset + frame_length] += weight[:frame_length]
        offset += stride
    assert sum_weight.min() > 0
    return out / sum_weight


def _get_checkpoint_url(root_url: str, checkpoint: str):
    if not root_url.endswith('/'):
        root_url += '/'
    return root_url + checkpoint


def _check_checksum(path: Path, checksum: str):
    sha = sha256()
    with open(path, 'rb') as file:
        while True:
            buf = file.read(2**20)
            if not buf:
                break
            sha.update(buf)
    actual_checksum = sha.hexdigest()[:len(checksum)]
    if actual_checksum != checksum:
        raise RuntimeError(f'Invalid checksum for file {path}, '
                           f'expected {checksum} but got {actual_checksum}')


def convert_audio(wav: torch.Tensor, sr: int, target_sr: int, target_channels: int):
    assert wav.dim() >= 2, "Audio tensor must have at least 2 dimensions"
    assert wav.shape[-2] in [1, 2], "Audio must be mono or stereo."
    *shape, channels, length = wav.shape
    if target_channels == 1:
        wav = wav.mean(-2, keepdim=True)
    elif target_channels == 2:
        wav = wav.expand(*shape, target_channels, length)
    elif channels == 1:
        wav = wav.expand(target_channels, -1)
    else:
        raise RuntimeError(f"Impossible to convert from {channels} to {target_channels}")
    wav = torchaudio.transforms.Resample(sr, target_sr)(wav)
    return wav


def save_audio(wav: torch.Tensor, path: tp.Union[Path, str],
               sample_rate: int, rescale: bool = False):
    """save audio

    Args:
        wav (torch.Tensor): Audio that needs to be saved
        path (tp.Union[Path, str]): storage path
        sample_rate (int): sample rate
        rescale (bool, optional): _description_. Defaults to False.
    """
    limit = 0.99
    mx = wav.abs().max()
    if rescale:
        wav = wav * min(limit / mx, 1)
    else:
        wav = wav.clamp(-limit, limit)
    torchaudio.save(str(path), wav, sample_rate=sample_rate, encoding='PCM_S', bits_per_sample=16)

def set_seed(seed):
    """set seed

    Args:
        seed (int): seed number
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def save_master_checkpoint(epoch, model, optimizer, scheduler, ckpt_name):  
    """save master checkpoint

    Args:
        epoch (int): epoch number
        model (nn.Module): model
        optimizer (optimizer): optimizer
        scheduler (_type_): _description_
        ckpt_name (str): checkpoint name
    """
    state_dict = {  
        'epoch': epoch,  
        'model_state_dict': model.state_dict(),  
        'optimizer_state_dict': optimizer.state_dict(),  
        'scheduler_state_dict': scheduler.state_dict(),  
    }  
    torch.save(state_dict, ckpt_name) 

def start_dist_train(train_fn, world_size, config, dist_init_method=None):  
    """start distribustion training

    Args:
        train_fn (_type_): train function
        world_size (_type_): world size
        config (_type_): config 
        dist_init_method (_type_, optional): dist init method. Defaults to None.
    """
    torch.multiprocessing.set_start_method('spawn')  
    mp.spawn(  
        train_fn,  
        args=(world_size, config, dist_init_method) if dist_init_method else (world_size, config, ),  
        nprocs=world_size,  
        join=True  
    )  

def count_parameters(model):  
    """count model parameters

    Args:
        model (nn.Module): model

    Returns:
        _type_: the model's parameters which are requires_grad
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)  

def collect_audio_durations(csv_files,output_path='./audio_durations.csv'):
    """analyse audio time durations

    Args:
        csv_files (.csv): generated train/test csv files
        output_path (str, optional): storage path. Defaults to './audio_durations.csv'.
    """
    audio_files = []
    if csv_files is not None:
        with open(csv_files,'r') as f:
            audio_files = [line.strip() for line in f.readlines()]
    durations = []
    for file in audio_files:  
        info = sf.info(file)  
        duration = info.duration 
        durations.append(duration)
    duration_data = pd.DataFrame({'filename': audio_files, 'duration': durations})
    duration_data.to_csv(output_path, index=False)  

def plot_audio_durations(duration_csv, boundaries, output_filename='audio_durations.png'):  
    import matplotlib.pyplot as plt
    duration_csv = Path(duration_csv)  
    assert duration_csv.exists(), "duration_csv isn't exists, need to use collect_audio_durations()"  
    # 读取音频时长数据  
    duration_data = pd.read_csv(duration_csv)  
    # 计算直方图的分割点  
    max_duration = int(np.ceil(duration_data['duration'].max()))  
    bins = np.arange(0, max_duration + 1, 1)  
    plt.figure(figsize=(12,5))  
    # 设置刻度字体大小  
    plt.rcParams['xtick.labelsize'] = 8  
    plt.rcParams['ytick.labelsize'] = 8  
    # 绘制直方图  
    counts, edges, patches = plt.hist(duration_data['duration'], bins=bins)  
    # 设置图像标题和轴标签  
    plt.title('audio durations distribution')  
    plt.xlabel('time(s)')  
    plt.ylabel('nums')  
    plt.xticks(np.arange(0, max_duration + 1, 1))  
    # 计算累积频率  
    cum_counts = np.cumsum(counts)  
    total_count = len(duration_data)  
    # 颜色列表  
    colors = ['r', 'g', 'b', 'm', 'c', 'y', 'k']  
    # 绘制分界线  
    for i, boundary in enumerate(boundaries):  
        percentage = total_count * boundary  
        idx = np.where(cum_counts >= percentage)[0][0]  
        color = colors[i % len(colors)]  # 从颜色列表中选择颜色  
        plt.axvline(x=edges[idx], color=color, linestyle='--', label=f'{int(boundary * 100)}%')  
    # 添加图例  
    plt.legend()  
    # 保存图像到文件  
    plt.savefig(output_filename, dpi=600)  
    # 显示图像  
    plt.show()
    