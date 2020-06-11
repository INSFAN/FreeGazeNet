import cv2
import numpy as np
import math

def calculate_pitch_yaw_roll(landmarks_2D, cam_w=256, cam_h=256,
                             radians=False):
    """ Return the the pitch  yaw and roll angles associated with the input image.
    @param radians When True it returns the angle in radians, otherwise in degrees.
    """

    assert landmarks_2D is not None, 'landmarks_2D is None'

    # Estimated camera matrix values.
    c_x = cam_w / 2
    c_y = cam_h / 2
    f_x = c_x / np.tan(60 / 2 * np.pi / 180)
    f_y = f_x
    camera_matrix = np.float32([[f_x, 0.0, c_x], [0.0, f_y, c_y],
                                [0.0, 0.0, 1.0]])
    camera_distortion = np.float32([0.0, 0.0, 0.0, 0.0, 0.0])

    # dlib (68 landmark) trached points
    # TRACKED_POINTS = [17, 21, 22, 26, 36, 39, 42, 45, 31, 35, 48, 54, 57, 8]
    # wflw(98 landmark) trached points
    # TRACKED_POINTS = [33, 38, 50, 46, 60, 64, 68, 72, 55, 59, 76, 82, 85, 16]
    # X-Y-Z with X pointing forward and Y on the left and Z up.
    # The X-Y-Z coordinates used are like the standard coordinates of ROS (robotic operative system)
    # OpenCV uses the reference usually used in computer vision:
    # X points to the right, Y down, Z to the front
    landmarks_3D = np.float32([
        [6.825897, 6.760612, 4.402142],  # LEFT_EYEBROW_LEFT, 
        [1.330353, 7.122144, 6.903745],  # LEFT_EYEBROW_RIGHT, 
        [-1.330353, 7.122144, 6.903745],  # RIGHT_EYEBROW_LEFT,
        [-6.825897, 6.760612, 4.402142],  # RIGHT_EYEBROW_RIGHT,
        [5.311432, 5.485328, 3.987654],  # LEFT_EYE_LEFT,
        [1.789930, 5.393625, 4.413414],  # LEFT_EYE_RIGHT,
        [-1.789930, 5.393625, 4.413414],  # RIGHT_EYE_LEFT,
        [-5.311432, 5.485328, 3.987654],  # RIGHT_EYE_RIGHT,
        [-2.005628, 1.409845, 6.165652],  # NOSE_LEFT,
        [-2.005628, 1.409845, 6.165652],  # NOSE_RIGHT,
        [2.774015, -2.080775, 5.048531],  # MOUTH_LEFT,
        [-2.774015, -2.080775, 5.048531],  # MOUTH_RIGHT,
        [0.000000, -3.116408, 6.097667],  # LOWER_LIP,
        [0.000000, -7.415691, 4.070434],  # CHIN
    ])
    landmarks_2D = np.asarray(landmarks_2D, dtype=np.float32).reshape(-1, 2)

    # Applying the PnP solver to find the 3D pose of the head from the 2D position of the landmarks.
    # retval - bool
    # rvec - Output rotation vector that, together with tvec, brings points from the world coordinate system to the camera coordinate system.
    # tvec - Output translation vector. It is the position of the world origin (SELLION) in camera co-ords
    _, rvec, tvec = cv2.solvePnP(landmarks_3D, landmarks_2D,
                                      camera_matrix, camera_distortion)
    #Get as input the rotational vector, Return a rotational matrix

    # const double PI = 3.141592653;
    # double thetaz = atan2(r21, r11) / PI * 180;
    # double thetay = atan2(-1 * r31, sqrt(r32*r32 + r33*r33)) / PI * 180;
    # double thetax = atan2(r32, r33) / PI * 180;
    
    rmat, _ = cv2.Rodrigues(rvec)
    pose_mat = cv2.hconcat((rmat, tvec))
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)
    return map(lambda k: k[0], euler_angles) # euler_angles contain (pitch, yaw, roll)

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def get_avg(self):
        return self.avg
    
    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, *meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def print(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

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
    gaze_pred_n_3d = np.array([-math.cos(gaze_pred[0]) * math.sin(gaze_pred[1]),
                              -math.sin(gaze_pred[0]),
                              -math.cos(gaze_pred[0]) * math.cos(gaze_pred[1])])
    gaze_n_3d_g = np.array([-math.cos(gaze_norm_g[0]) * math.sin(gaze_norm_g[1]),
                              -math.sin(gaze_norm_g[0]),
                              -math.cos(gaze_norm_g[0]) * math.cos(gaze_norm_g[1])])
    
    # convet rotation vector to rotation matrix
    rot_mat_norm, _ = cv2.Rodrigues(rot_vec_norm)

    # convert vector from normalization space to camera coordinate system
    gaze_pred_cam = np.linalg.inv(rot_mat_norm).dot(gaze_pred_n_3d.reshape(3, 1)) 
    gaze_pred_cam /= np.linalg.norm(gaze_pred_cam)

    gaze_g_cam = np.linalg.inv(rot_mat_norm).dot(gaze_n_3d_g.reshape(3, 1))
    gaze_g_cam /= np.linalg.norm(gaze_g_cam)

    error = np.arccos(gaze_pred_cam.T.dot(gaze_g_cam))

    # gaze_pred_cam_pitchyaw = vector_to_pitchyaw(-gaze_pred_cam.T).flatten()
    # gaze_g_cam_pitchyaw = vector_to_pitchyaw(-gaze_g_cam.T).flatten()

    return error * 180.0 / np.pi

def mean_angle_error(gaze_preds, gaze_norm_gs, rot_vec_norms):
    n = gaze_preds.shape[0]
    mean_error = []
    for i in range(n):
        mean_error.append(angle_error(gaze_preds[i], gaze_norm_gs[i], rot_vec_norms[i]))

    return np.mean(mean_error)




    

    