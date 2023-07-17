import os, pickle, torch
import numpy as np


if os.path.exists("D:\Yuanbo\Dataset\DCASE2018_T1_A"):
    dataset_dir="D:\Yuanbo\Dataset\DCASE2018_T1_A"

elif os.path.exists("E:\yuanbo\Dataset\DCASE2018_T1_A"):
    dataset_dir="E:\yuanbo\Dataset\DCASE2018_T1_A"

elif os.path.exists("/home/hyb/.data"):
    dataset_dir="/home/hyb/.data"

elif os.path.exists("/project_antwerp/yuanbo/Dataset/DCASE2018_T1_A"):
    dataset_dir="/project_antwerp/yuanbo/Dataset/DCASE2018_T1_A"

elif os.path.exists("/project_scratch/yuanbo/Dataset/DCASE2018_T1_A"):
    dataset_dir="/project_scratch/yuanbo/Dataset/DCASE2018_T1_A"

elif os.path.exists("/project/yuanbo/Dataset/DCASE2018_T1_A"):
    dataset_dir="/project/yuanbo/Dataset/DCASE2018_T1_A"


subdir = os.path.join(dataset_dir, 'TUT-urban-acoustic-scenes-2018-development')
if not os.path.exists(subdir):
    subdir = os.path.join(dataset_dir, 'development')

holdout_fold = 1
# 2021-9-7 add
cuda_seed = None  # 1024
cuda = 1

training = 1
testing = 1

if cuda:
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
else:
    device = torch.device('cpu')


sample_rate = 44100
window_size = 2048
overlap = 672   # So that there are 320 frames in an audio clip
seq_len = 320
mel_bins = 64

epochs = 10000  # 8000  # 5000  # 3230  # 5000  # 10000
batch_size = 64  # 100

only_save_best = True

devices = ['a']

labels = ['airport', 'bus', 'metro', 'metro_station', 'park', 'public_square',
          'shopping_mall', 'street_pedestrian', 'street_traffic', 'tram']

lb_to_ix = {lb: ix for ix, lb in enumerate(labels)}
ix_to_lb = {ix: lb for ix, lb in enumerate(labels)}

endswith = '.pth'


