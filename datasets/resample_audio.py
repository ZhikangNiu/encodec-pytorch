import argparse
from pathlib import Path

import torch
import torchaudio
from tqdm import tqdm


def get_parser():
    parser = argparse.ArgumentParser(description="Convert sample rate of all audio files in source_dir and saves to target_dir")
    parser.add_argument(
        '-s',
        '--source_dir',
        required=True,
        help="Source wave folder."
    )
    parser.add_argument(
        '-t',
        '--target_sr',
        type=int,
        default=24000,
        help="Target sample rate."
    )
    parser.add_argument(
        '-c',
        '--target_channels',
        type=int,
        default=1,
        help="Target channels."
    )
    parser.add_argument(
        '-e',
        '--file_extension',
        type=str,
        default='flac',
        help="File extension.",
        choices=['flac','wav']
    )
    return parser

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

def convert_sample_rate(source_dir,target_sr=24000,target_channels=1,file_extension='.flac'):
    """Converts sample rate of all audio files in source_dir and saves to target_dir"""
    source_dir = Path(source_dir)
    target_dir = source_dir.parent / f"{source_dir.name}_{int(target_sr/1000)}khz"
    for wav_path in tqdm(list(source_dir.rglob(f'*.{file_extension}'))):
        relative_path = wav_path.relative_to(source_dir)
        wav,sr = torchaudio.load(wav_path)  # Load audio
        resample_wav = convert_audio(wav,sr,target_sr,target_channels)
        save_path = target_dir / relative_path
        if not save_path.parent.exists():
            save_path.parent.mkdir(parents=True)
        torchaudio.save(save_path, resample_wav, sample_rate=target_sr)

def main():
    parser = get_parser()
    args = parser.parse_args()
    print(args)
    convert_sample_rate(args.source_dir,args.target_sr,args.target_channels,args.file_extension)

if __name__ == "__main__":
    main()