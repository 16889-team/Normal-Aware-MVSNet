import argparse
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import time
from tensorboardX import SummaryWriter
from datasets import find_dataset_def
from models import *
from utils import *
import gc
import sys
import datetime


# # dataset, dataloader
# MVSDataset = find_dataset_def('dtu_yao')
# train_dataset = MVSDataset( "datasets/mvs_training/eth/forest", 'lists/dtu/train.txt', 'train', 3, 128)
# item = train_dataset[50]['imgs']

# # test homography here
# print(item.keys())
# print("imgs", item["imgs"].shape)
# print("depth", item["depth"].shape)
# print("depth_values", item["depth_values"].shape)
# print("mask", item["mask"].shape)

# ref_img = item["imgs"][0].transpose([1, 2, 0])[::4, ::4]
# src_imgs = [item["imgs"][i].transpose([1, 2, 0])[::4, ::4] for i in range(1, 5)]
# ref_proj_mat = item["proj_matrices"][0]
# src_proj_mats = [item["proj_matrices"][i] for i in range(1, 5)]
# mask = item["mask"]
# depth = item["depth"]

# height = ref_img.shape[0]
# width = ref_img.shape[1]
# xx, yy = np.meshgrid(np.arange(0, width), np.arange(0, height))
# print("yy", yy.max(), yy.min())
# yy = yy.reshape([-1])
# xx = xx.reshape([-1])
# X = np.vstack((xx, yy, np.ones_like(xx)))
# D = depth.reshape([-1])
# print("X", "D", X.shape, D.shape)

# X = np.vstack((X * D, np.ones_like(xx)))
# X = np.matmul(np.linalg.inv(ref_proj_mat), X)
# X = np.matmul(src_proj_mats[0], X)
# X /= X[2]
# X = X[:2]

# yy = X[0].reshape([height, width]).astype(np.float32)
# xx = X[1].reshape([height, width]).astype(np.float32)
# import cv2

# warped = cv2.remap(src_imgs[0], yy, xx, interpolation=cv2.INTER_LINEAR)
# warped[mask[:, :] < 0.5] = 0

# cv2.imwrite('../tmp0.png', ref_img[:, :, ::-1] * 255)
# cv2.imwrite('../tmp1.png', warped[:, :, ::-1] * 255)
# cv2.imwrite('../tmp2.png', src_imgs[0][:, :, ::-1] * 255)

