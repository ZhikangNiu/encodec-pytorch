import os

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
    file_dir = '/mnt/lustre/sjtu/home/zkn02/data/LibriSpeech/train-clean-100'
    csv_path = './librispeech_train100h.csv'
    generate_csv(file_dir, csv_path)