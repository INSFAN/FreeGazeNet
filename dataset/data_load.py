import numpy as np
import sys
sys.path.append('..')
import h5py
from torch.utils import data
from torch.utils.data import DataLoader
import cv2

class MPIIDatasets(data.Dataset):
    def __init__(self, file_path, train=True, transforms=None):
        self.transforms = transforms
        self.person_id = None
        self.groups = []
        self.num_entries = 0
        self.train = train
        if (self.train):
            self.dataset_pixels =  h5py.File(file_path, 'r')['pixels'][:-1500]
            self.dataset_labels =  h5py.File(file_path, 'r')['labels'][:-1500]
        else:
            self.dataset_pixels =  h5py.File(file_path, 'r')['pixels'][-1500:]
            self.dataset_labels =  h5py.File(file_path, 'r')['labels'][-1500:]


    def __getitem__(self, index):
        self.patch = self.dataset_pixels[index]
        self.gaze_norm_g = self.dataset_labels[index][0:2].flatten()
        self.head_norm =self.dataset_labels[index][2:4].flatten()
        self.rot_vec_norm = self.dataset_labels[index][4:7].flatten()

        if self.transforms:
            self.patch = self.transforms(self.patch)
        return (self.patch, self.gaze_norm_g, self.head_norm, self.rot_vec_norm) 

    def __len__(self):
        return self.dataset_labels.shape[0]

def vector_to_pitchyaw(vectors):
    """Convert given gaze vectors to yaw (theta) and pitch (phi) angles."""
    '''x = cos(yaw)cos(pitch) y = sin(yaw)cos(pitch) z = sin(pitch)'''
    n = vectors.shape[0]
    out = np.empty((n, 2))
    vectors = np.divide(vectors, np.linalg.norm(vectors, axis=1).reshape(n, 1))
    out[:, 0] = np.arcsin(vectors[:, 1])  # theta
    out[:, 1] = np.arctan2(vectors[:, 0], vectors[:, 2])  # phi
    return out

def angle_error(gaze_pred, gaze_norm_g, rot_vec_norm):
    # convert ptich yam to 3d x, y, z in normalization space
    gaze_pred_n_3d = np.array([np.cos(gaze_pred[0]) * np.sin(gaze_pred[1]),
                              np.sin(gaze_pred[0]),
                              np.cos(gaze_pred[0]) * np.cos(gaze_pred[1])])
    gaze_n_3d_g = np.array([np.cos(gaze_norm_g[0]) * np.sin(gaze_norm_g[1]),
                              np.sin(gaze_norm_g[0]),
                              np.cos(gaze_norm_g[0]) * np.cos(gaze_norm_g[1])])
    
    # convet rotation vector to rotation matrix
    rot_mat_norm, _ = cv2.Rodrigues(rot_vec_norm)

    # convert vector from normalization space to camera coordinate system
    gaze_pred_cam = np.linalg.inv(rot_mat_norm) * gaze_pred_n_3d
    gaze_pred_cam /= np.linalg.norm(gaze_pred_cam)

    gaze_g_cam = np.linalg.inv(rot_mat_norm) * gaze_n_3d_g
    gaze_g_cam /= np.linalg.norm(gaze_g_cam)

    gaze_pred_cam_pitchyaw = vector_to_pitchyaw(-gaze_pred_cam.T).flatten()
    gaze_g_cam_pitchyaw = vector_to_pitchyaw(-gaze_g_cam.T).flatten()

    return np.mean(np.fabs(gaze_pred_cam_pitchyaw - gaze_g_cam_pitchyaw))

def mean_angle_error(gaze_preds, gaze_norm_gs, rot_vec_norms):
    n = gaze_preds.shape[0]
    mean_error = 0.0
    for i in range(n):
        mean_error += angle_error(gaze_preds[i], gaze_norm_gs[i], rot_vec_norms[i])

    return mean_error / n

if __name__ == '__main__':
    file_path = './data/MPIIGaze_448.h5'
    mpiidataset = MPIIDatasets(file_path, train=True, transforms=None)
    dataloader = DataLoader(mpiidataset, batch_size=32, shuffle=True, num_workers=0, drop_last=False)
    print(len(mpiidataset))
    for patch, gaze_norm_g, head_norm, rot_vec_norm in dataloader:
        print("img shape", patch.shape)
        head_norm = head_norm.numpy()
        gaze_norm_g = gaze_norm_g.numpy()
        rot_vec_norm = rot_vec_norm.numpy()
        print("gaze_norm_g :", gaze_norm_g.shape)
        print("head_norm :", head_norm.shape)
        print("rot_vec_norm", rot_vec_norm.shape)
        print(mean_angle_error(head_norm, gaze_norm_g, rot_vec_norm))
        break



# transform = transforms.Compose([
#     transforms.RandomCrop(32, padding=4),  # 先四周填充0，在把图像随机裁剪成32*32
#     transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
#     transforms.RandomRotation((-45, 45)),  # 随机旋转
#     transforms.ToTensor(),
#     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.229, 0.224, 0.225))
#     # R, G, B每层的归一化用到的均值和方差
# ])
