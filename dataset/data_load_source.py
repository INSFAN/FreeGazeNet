import numpy as np
import sys
sys.path.append('..')
import h5py
from torch.utils import data
from torch.utils.data import DataLoader
from data_utils import data_normalization

class WLFWDatasets(data.Dataset):
    def __init__(self, file_list, transforms=None):
        self.line = None
        self.path = None
        self.landmarks = None
        self.attribute = None
        self.filenames = None
        self.euler_angle = None
        self.transforms = transforms
            # Grab SFM coordinates and store
        self.face_model_fpath = '../data/sfm_face_coordinates.npy'
        self.input_path = '/media/insfan/00028D8D000E9194/MPIIFaceGaze/MPIIFaceGaze'
        self.supplementary =  '../data/MPIIFaceGaze_supplementary.h5'
        self.face_model_3d_coordinates = np.load(self.face_model_fpath)
        self.person_id = None
        self.groups = []
        self.num_entries = 0
        with h5py.File(self.supplementary, 'r') as f:
            for self.person_id, group in f.items():
                self.groups.append(group)
                self.num_entries = self.num_entries + next(iter(group.values())).shape[0]

        # with open(file_list, 'r') as f:
        #     self.lines = f.readlines()

    def __getitem__(self, index):
        self.patch, self.n_gaze, self.n_head, self.n_rot_vec = data_normalization('MPIIGaze',
                                   self.input_path,
                                   self.groups,
                                   self.face_model_3d_coordinates,
                                   index)
        if self.transforms:
            self.patch = self.transforms(self.patch)
        return (self.patch, self.n_gaze, self.n_head, self.n_rot_vec) 

    def __len__(self):
        self.num_entries = 0
        for i in range(len(self.groups)):
            self.num_entries += next(iter(self.groups[i].values())).shape[0]
        return self.num_entries

if __name__ == '__main__':
    # file_list = './data/test_data/list.txt'
    wlfwdataset = WLFWDatasets(None)
    dataloader = DataLoader(wlfwdataset, batch_size=1, shuffle=True, num_workers=0, drop_last=False)
    for patch, n_gaze, n_head, n_rot_vec in dataloader:
        print("img shape", patch.shape)
        print("n_gaze :", n_gaze)
        print("n_head :", n_head)
        print("n_rot_vec", n_rot_vec)



# transform = transforms.Compose([
#     transforms.RandomCrop(32, padding=4),  # 先四周填充0，在把图像随机裁剪成32*32
#     transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
#     transforms.RandomRotation((-45, 45)),  # 随机旋转
#     transforms.ToTensor(),
#     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.229, 0.224, 0.225))
#     # R, G, B每层的归一化用到的均值和方差
# ])