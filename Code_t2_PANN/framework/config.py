import os, pickle, torch, csv
import numpy as np


if os.path.exists("D:\Yuanbo\Dataset\DCASE2018_T1_A"):
    dataset_dir="D:\Yuanbo\Dataset\DCASE2018_T1_A"

elif os.path.exists("E:\yuanbo\Dataset\DCASE2018_T1_A"):
    dataset_dir="E:\yuanbo\Dataset\DCASE2018_T1_A"

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
dev_train_csv = os.path.join(subdir, 'evaluation_setup', 'fold{}_train.txt'.format(holdout_fold))
dev_validate_csv = os.path.join(subdir, 'evaluation_setup', 'fold{}_validation.txt'.format(holdout_fold))
dev_evaluate_csv = os.path.join(subdir, 'evaluation_setup', 'fold{}_evaluate.txt'.format(holdout_fold))
cuda_seed = None  # 1024
cuda = 1

lr_init = 1e-3
batch_size = 64
epochs = 100
#################### pretrain model ########################################################
if cuda:
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
else:
    device = torch.device('cpu')

labels = ['airport', 'bus', 'metro', 'metro_station', 'park', 'public_square',
          'shopping_mall', 'street_pedestrian', 'street_traffic', 'tram']

lb_to_ix = {lb: ix for ix, lb in enumerate(labels)}
ix_to_lb = {ix: lb for ix, lb in enumerate(labels)}

endswith = '.pth'

