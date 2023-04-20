import os
import shutil
import torch
import torchaudio

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

def convert_sample_rate(source_dir,target_sr=24000,target_channels=1):
    """Converts sample rate of all audio files in source_dir and saves to target_dir"""
    dataset_name = os.path.basename(source_dir)
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.endswith('.flac') or file.endswith('.wav'):
                file_path = os.path.join(root, file)
                wav,sr = torchaudio.load(file_path)  # Load audio
                resample_wav = convert_audio(wav,sr,target_sr,target_channels)
                if not os.path.exists(root.replace(dataset_name, f"{dataset_name}_{int(target_sr/1000)}khz")):
                    print(root.replace(dataset_name, f"{dataset_name}_{int(target_sr/1000)}khz"))
                    os.makedirs(root.replace(dataset_name, f"{dataset_name}_{int(target_sr/1000)}khz"))
                save_path = file_path.replace(dataset_name, f"{dataset_name}_{int(target_sr/1000)}khz")
                # shutil.copy(filename, os.path.join(target_dir, file))
                torchaudio.save(save_path, resample_wav, target_sr,target_channels)  # Save resampled
            
        
if __name__ == "__main__":
    import argparse
    args = argparse.ArgumentParser()
    args.add_argument("--source_dir", type=str, default="datasets")
    args = args.parse_args()
    convert_sample_rate(args.source_dir) 