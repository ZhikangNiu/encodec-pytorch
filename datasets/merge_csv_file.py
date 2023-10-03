import argparse

import pandas as pd


def merge_file_main(args):
    """merge many csv file

    Args:
        args (_type_): args
    """
    merge_file_list = []
    for csv_file in args.inputs:
        data = pd.read_csv(csv_file,header=None,on_bad_lines='skip')
        merge_file_list.append(data)
    
    merged_file = pd.concat(merge_file_list, ignore_index=True)  
      
    merged_file.to_csv(args.output_path, index=False, header=None)  

if __name__ == "__main__":
    arg = argparse.ArgumentParser()
    arg.add_argument('-i', '--inputs', nargs='+', help='input1.csv input2.csv input3.csv', required=True) 
    arg.add_argument('-o','--output_path', type=str, default='./merged.csv')
    args = arg.parse_args()
    merge_file_main(args)