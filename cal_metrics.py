# core codes are copy from https://github.com/yangdongchao/AcademiCodec/tree/master/evaluation_metric/calculate_voc_obj_metrics/metrics
import argparse
from pesq import pesq,cypesq
from pystoi import stoi
from pathlib import Path
import librosa
import numpy as np
from tqdm import tqdm
from audiotools.metrics.quality import visqol
def get_parser():
    parser = argparse.ArgumentParser(description="Compute STOI and PESQ measure")
    parser.add_argument(
        '-r',
        '--ref_dir',
        required=True,
        help="Reference wave folder."
    )
    parser.add_argument(
        '-d',
        '--deg_dir',
        required=True,
        help="Degraded wave folder."
    )
    parser.add_argument(
        '-s',
        '--sr',
        type=int,
        default=24000,
        help="encodec sample rate."
    )
    parser.add_argument(
        '-b',
        '--bandwidth',
        type=float,
        default=6.0,
        help="encodec bandwidth.",
        choices=[1.5, 3.0, 6.0, 12.0, 24.0, 48.0]
    )
    return parser


def calculate_stoi(ref_wav, deg_wav, sr):
    """Calculate STOI score between ref_wav and deg_wav"""
    min_len = min(len(ref_wav), len(deg_wav))
    ref_wav = ref_wav[:min_len]
    deg_wav = deg_wav[:min_len]
    stoi_score = stoi(ref_wav, deg_wav, sr, extended=False)
    return stoi_score

def calculate_pesq(ref_wav, deg_wav, sr):
    """Calculate PESQ score between ref_wav and deg_wav, we need to resample to 16000Hz first"""
    min_len = min(len(ref_wav), len(deg_wav))
    ref_wav = ref_wav[:min_len]
    deg_wav = deg_wav[:min_len]
    nb_pesq_score = pesq(sr, ref_wav, deg_wav, 'nb')
    wb_pesq_score = pesq(sr, ref_wav, deg_wav, 'wb')
    return nb_pesq_score, wb_pesq_score

def calculate_visqol(ref_wav,deg_wav,mode='audio'):
    pass
def main():
    args = get_parser().parse_args()
    stoi_scores = []
    nb_pesq_scores = []
    wb_pesq_scores = []
    for deg_wav_path in tqdm(list(Path(args.deg_dir).rglob('*.wav'))):
        relative_path = deg_wav_path.relative_to(args.deg_dir)
        ref_wav_path = Path(args.ref_dir) / relative_path.parents[0] /deg_wav_path.name.replace(f'_bw{args.bandwidth}', '')
        ref_wav,_ = librosa.load(ref_wav_path, sr=args.sr)
        deg_wav,_ = librosa.load(deg_wav_path, sr=args.sr)
        stoi_score = calculate_stoi(ref_wav, deg_wav, sr=args.sr)
        if args.sr != 16000:
            ref_wav = librosa.resample(y=ref_wav, orig_sr=args.sr, target_sr=16000)
            deg_wav = librosa.resample(y=deg_wav, orig_sr=args.sr, target_sr=16000)
        try:
            nb_pesq_score, wb_pesq_score = calculate_pesq(ref_wav, deg_wav, 16000)
            nb_pesq_scores.append(nb_pesq_score)
            wb_pesq_scores.append(wb_pesq_score)
        except cypesq.NoUtterancesError:
            print(ref_wav_path)
            print(deg_wav_path)
            nb_pesq_score, wb_pesq_score = 0, 0
        if stoi_score!=1e-5:
            stoi_scores.append(stoi_score)
    return np.mean(stoi_scores), np.mean(nb_pesq_scores), np.mean(wb_pesq_scores)
if __name__ == '__main__':
    mean_stoi, mean_nb_pesq, mean_wb_pesq = main()
    print(f"STOI: {mean_stoi}")
    print(f"NB PESQ: {mean_nb_pesq}")
    print(f"WB PESQ: {mean_wb_pesq}")


