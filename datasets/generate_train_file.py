import os
import argparse

def generate_csv(file_dir, csv_path):
    # 生成file_dir下所有文件的路径
    file_list = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if file.endswith('.flac') and 'train' in root:
                file_list.append(os.path.join(root, file))
    # 生成csv文件
    with open(csv_path, 'w') as f:
        for file in file_list:
            f.write(file + '\n')


if __name__ == '__main__':
    arg = argparse.ArgumentParser()
    arg.add_argument('-i','--input_file_dir', type=str, default='./LibriSpeech/train-clean-100')
    arg.add_argument('-o,''--output_path', type=str, default='./librispeech_train100h.csv')
    args = arg.parse_args()
    assert args.file_dir is not None and os.path.exists(args.file_dir), 'file_dir is not exist'
    generate_csv(args.input_file_dir, args.output_path)