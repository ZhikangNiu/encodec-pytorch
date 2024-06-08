import os
import argparse
from pathlib import Path
import pandas as pd  

def generate_csv(file_dir, csv_path,mode='train'):
    # 生成file_dir下所有文件的路径
    file_list = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if (file.endswith('.flac') or file.endswith('.wav') or file.endswith('.mp3')) and mode in root:
                file_list.append(os.path.join(root, file))
    print(f"file length:{len(file_list)}")
    # 生成csv文件
    csv_path = Path(csv_path)
    if not csv_path.parent.exists():
        csv_path.parent.mkdir(parents=True)
        
    data = pd.DataFrame(file_list)
    data.to_csv(csv_path, index=False, header=False)

def split_train_test_csv(csv_path, threshold=0.8):
    try:
        from sklearn.model_selection import train_test_split  
    except ImportError as E:
        print("please pip install pandas slearn")
        
    data = pd.read_csv(csv_path)  
    train_data, test_data = train_test_split(data, train_size=threshold, random_state=42)  
   
    train_data.to_csv(f'{Path(csv_path).stem}_train.csv', index=False)  
    test_data.to_csv(f'{Path(csv_path).stem}_test.csv', index=False) 

if __name__ == '__main__':
    arg = argparse.ArgumentParser()
    arg.add_argument('-i','--input_file_dir', type=str, default='./LibriSpeech/train-clean-100')
    arg.add_argument('-o','--output_path', type=str, default='./librispeech_train100h.csv')
    arg.add_argument('-m','--mode', type=str, default='train',help='train,test-clean/other or dev-clean/other')
    arg.add_argument('-s','--split', action='store_true', default=False,help='slpit dataset')
    arg.add_argument('-t','--threshold',type=float,default=0.8)
    args = arg.parse_args()
    generate_csv(args.input_file_dir, args.output_path,args.mode)
    if args.split:
        split_train_test_csv(args.output_path,threshold=args.threshold)