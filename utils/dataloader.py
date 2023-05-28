from random import randint

import cv2
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from .utils import cvtColor, preprocess_input


def get_new_img_size(width, height, img_min_side=600):
    if width <= height:
        f = float(img_min_side) / width
        resized_height = int(f * height)
        resized_width = int(img_min_side)
    else:
        f = float(img_min_side) / height
        resized_width = int(f * width)
        resized_height = int(img_min_side)

    return resized_width, resized_height

class SRGANDataset(Dataset):
    def __init__(self,  lr_shape, hr_shape):
        super(SRGANDataset, self).__init__()

        import pickle

        with open("data.pkl","rb") as f:
            self.data = pickle.load(f)
        with open("noiseData.pkl","rb") as f:
            self.data_l = pickle.load(f)

        self.length             = self.data.shape[0]
    
        self.lr_shape           = lr_shape
        self.hr_shape           = hr_shape

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        index = index % self.length
        
        img_h = self.data[index,:,:,:]
        
        img_l =self.data_l[index,:,:,:]
        in_channels_h=img_h.shape[-1]
        in_channels_l=img_l.shape[-1]
        

        # pytorch  图片通道在前
        img_h = np.transpose(preprocess_input(np.array(img_h, dtype=np.float32)), [2, 0, 1])
        img_l = np.transpose(preprocess_input(np.array(img_l, dtype=np.float32)), [2, 0, 1])

        img_h = torch.from_numpy(img_h.reshape(1,img_h.shape[0],img_h.shape[1],img_h.shape[-1]))
        img_l = torch.from_numpy(img_l.reshape(1,img_l.shape[0],img_l.shape[1],img_l.shape[-1]))
        

        convh = nn.Conv2d(in_channels=in_channels_h,out_channels=3,kernel_size=3,stride=1,bias=True)
        convl = nn.Conv2d(in_channels=in_channels_l,out_channels=3,kernel_size=3,stride=1,bias=True)
        
        # 自定义卷积核
        # conv_kernel = np.array([
        #     [[[-1, 0, 1], [0, 0, 0], [1, -1, 1]],
        #     [[-1,1,1],[0,1,0],[1,0,0]],
        #     [[1,-1,1],[-1,1,0],[0,1,0]]],

        #     [[[0, 0, 1], [1, -1, 1], [0, 0, 1]],
        #     [[1,0,1],[-1,0,-1],[0,-1,0]],
        #     [[0,1,1],[-1,-1,0],[1,1,0]]],

        #     [[[1, 0, 1], [1, -1, 1], [0, 0, 1]],
        #     [[1,0,1],[-1,1,-1],[2,-1,0]],
        #     [[0,1,1],[-1,-1,1],[1,1,0]]],
        #     [[[1, 0, 1], [1, -1, 1], [0, 0, 1]],
        #     [[1,0,1],[-1,1,-1],[2,-1,0]],
        #     [[0,1,1],[-1,-1,1],[1,1,0]]]
        # ], dtype='float32')
        conv_kernel_h = np.ones(in_channels_h*3,dtype=np.float32)
        conv_kernel_l = np.ones(in_channels_l*3,dtype=np.float32)

        # 自定义偏置项
        conv_bias_h = np.array([0.,0.,0.],dtype=np.float32)
        conv_bias_l = np.array([0.,0.,0.],dtype=np.float32)

        # 适配卷积的输入输出
        conv_kernel_h = conv_kernel_h.reshape((3, in_channels_h, 1, 1))
        conv_kernel_l = conv_kernel_l.reshape((3, in_channels_h, 1, 1))

        convh.weight.data = torch.from_numpy(conv_kernel_h) 	# 给卷积的 kernel 赋值
        convh.bias.data  = torch.from_numpy(conv_bias_h)

        convl.weight.data = torch.from_numpy(conv_kernel_l) 	# 给卷积的 kernel 赋值
        convl.bias.data  = torch.from_numpy(conv_bias_l)


        # print(conv_kernel.shape)
        # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        # print(img_h.shape)
        # print("~~~~~~~~~~~~~~~~~-----------~~~~~~~~~~~~~~~~~~~~~~~~~")

        img_h = convh(Variable(img_h)).data.squeeze().numpy()
        img_l = convl(Variable(img_l)).data.squeeze().numpy()

        # print(img_h.shape,img_l.shape)#(3, 16, 32) (3, 16, 32)
        # print(type(img_l))
        # print("++++++++++++++++++++++++++++++++++++++++++")


        # img_l = np.resize(img_l,(self.lr_shape[0],self.lr_shape[1],3))

        # img_h = np.transpose(preprocess_input(np.array(img_h, dtype=np.float32)), [2, 0, 1])
        # img_l = np.transpose(preprocess_input(np.array(img_l, dtype=np.float32)), [2, 0, 1])

        return np.array(img_l)/in_channels_l/3, np.array(img_h)/in_channels_h/3

    def rand(self, a=0, b=1):
        return np.random.rand()*(b-a) + a


        
def SRGAN_dataset_collate(batch):
    images_l = []
    images_h = []
    for img_l, img_h in batch:
        images_l.append(img_l)
        images_h.append(img_h)
        
    images_l = torch.from_numpy(np.array(images_l, np.float32))
    images_h = torch.from_numpy(np.array(images_h, np.float32))
    return images_l, images_h
