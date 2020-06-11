#-*- coding:utf-8 -*-
import numpy as np
import cv2
import sys
sys.path.append('..')

from torch.utils import data
from torch.utils.data import DataLoader

class MPIIDatasets(data.Dataset):
    def __init__(self, file_list, train=True, transforms=None):
        self.line = None
        self.path = None
        self.gaze_norm_g = None
        self.head_norm = None
        self.filenames = None
        self.rot_vec_norm = None
        self.train = train
        self.transforms = transforms
        with open(file_list, 'r', encoding='utf-8') as f:
            if (self.train):
                self.lines = f.readlines()[:-1500]  # encoding='unicode_escape'
            else:
                self.lines = f.readlines()[-1500:]
        
    def __getitem__(self, index):
        self.line = self.lines[index].strip().split()
        self.img = cv2.imread(self.line[0])
        self.img = self.img[:, :, ::-1].copy()  # BGR to RGB
        self.gaze_norm_g = np.asarray(self.line[1:3], dtype=np.float32)
        self.head_norm = np.asarray(self.line[3:5], dtype=np.float32)
        self.rot_vec_norm = np.asarray(self.line[5:8], dtype=np.float32)
        if self.transforms:
            self.img = self.transforms(self.img)
        return (self.img, self.gaze_norm_g, self.head_norm, self.rot_vec_norm)

    def __len__(self):
        return len(self.lines)


class GazeCaptureDatasets(data.Dataset):
    def __init__(self, file_list, train=True, transforms=None):
        self.line = None
        self.path = None
        self.gaze_norm_g = None
        self.head_norm = None
        self.filenames = None
        self.rot_vec_norm = None
        self.train = train
        self.transforms = transforms
        with open(file_list, 'r', encoding='utf-8') as f:
            if (self.train):
                self.lines = f.readlines()[:-5000]  # encoding='unicode_escape'
            else:
                self.lines = f.readlines()[-5000:]
        
    def __getitem__(self, index):
        self.line = self.lines[index].strip().split()
        self.img = cv2.imread(self.line[0])
        self.img = self.img[:, :, ::-1].copy()  # BGR to RGB
        self.gaze_norm_g = np.asarray(self.line[1:3], dtype=np.float32)
        self.head_norm = np.asarray(self.line[3:5], dtype=np.float32)
        self.rot_vec_norm = np.asarray(self.line[5:8], dtype=np.float32)
        if self.transforms:
            self.img = self.transforms(self.img)
        return (self.img, self.gaze_norm_g, self.head_norm, self.rot_vec_norm)

    def __len__(self):
        return len(self.lines)


if __name__ == '__main__':
    file_list = './data/MPIIFaceGaze/norm_list.txt'
    mpiidataset = MPIIDatasets(file_list, train=False)
    dataloader = DataLoader(mpiidataset, batch_size=1, shuffle=True, num_workers=0, drop_last=False)
    print('len(dataloader): ', len(mpiidataset))
    for img, gaze_norm_g, head_norm, rot_vec_norm in dataloader:
        print("img shape : ", img.shape)
        print("gaze_norm_g: ", gaze_norm_g)
        print("head_norm : ", head_norm)
        print("rot_vec_norm : ", rot_vec_norm)
        break



# transform = transforms.Compose([
#     transforms.RandomCrop(32, padding=4),  # 先四周填充0，在把图像随机裁剪成32*32
#     transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
#     transforms.RandomRotation((-45, 45)),  # 随机旋转
#     transforms.ToTensor(),
#     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.229, 0.224, 0.225))
#     # R, G, B每层的归一化用到的均值和方差
# ])