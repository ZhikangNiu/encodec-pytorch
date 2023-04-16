# datasets
You need to use the `generate_train_file.py` script to generate the datasets train csv. 
The script will generate the csv files for the train datasets. The train csv is listed as follows:
```
# train_encodec/datasets/train.csv
YOUR_PATH/LibriSpeech/train-clean-100/7794/295955/7794-295955-0003.flac
YOUR_PATH/LibriSpeech/train-clean-100/7794/295955/7794-295955-0027.flac
YOUR_PATH/LibriSpeech/train-clean-100/7794/295955/7794-295955-0014.flac
YOUR_PATH/LibriSpeech/train-clean-100/7794/295955/7794-295955-0029.flac
YOUR_PATH/LibriSpeech/train-clean-100/7794/295955/7794-295955-0008.flac
YOUR_PATH/LibriSpeech/train-clean-100/7794/295955/7794-295955-0012.flac
YOUR_PATH/LibriSpeech/train-clean-100/7794/295955/7794-295955-0006.flac
YOUR_PATH/LibriSpeech/train-clean-100/7794/295955/7794-295955-0018.flac
```

the script get two arguments:
- `--input_file_dir`: the path of the LibriSpeech dataset
- `--output_path`: the path of the output csv file