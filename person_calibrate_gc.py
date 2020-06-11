#!/usr/bin/env python

'''
camera calibration for distorted images with chess board samples
reads distorted images, calculates the calibration and write undistorted images
usage:
    calibrate.py [--out <output path>] [--square_size] [<image mask>]
default values:
    --out:    .sample/output/
    --square_size: 1.0
    <image mask> defaults to .sample/chessboard/*.jpg


Code forked from OpenCV:
https://github.com/opencv/opencv/blob/a8e2922467bb5bb8809fc8ae6b7233893c6db917/samples/python/calibrate.py
released under BSD 3 license
'''

'''Screen size: DP-1 connected primary 1920x1080+0+0 (normal left inverted right x axis y axis) 477mm x 268mm
'''


# built-in modules
import os
import sys
from glob import glob
import numpy as np
import cv2
import logging
import argparse
import copy
import matplotlib.pyplot as plt  
from mpl_toolkits.mplot3d import Axes3D  # 空间三维画图
from sklearn import linear_model
import math
import random
from sklearn.preprocessing import PolynomialFeatures
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from statsmodels.iolib.table import (SimpleTable, default_txt_fmt)
import statsmodels.api as sm
from sklearn.neural_network import MLPRegressor
import json


logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

mpi_index = {'p00':(0, 2925), 'p01':(2925, 5828),'p02':(5828, 8743),'p03':(8743, 11666),'p04':(11666, 14526),
             'p05':(14526, 17396),'p06':(17396, 20273),'p07':(20273, 23113),'p08':(23113, 25846),'p09':(25846, 28558),
             'p10':(28558, 30736),'p11':(30736, 32998),'p12':(32998, 34591),'p13':(34591, 36077),'p14':(36077, 37577)}

def parse_args():
    # Parse arguments
    parser = argparse.ArgumentParser(description='get camera cailbrate images')
    parser.add_argument('--gaze_data', default='00135_norm_predict.npy', help='path to predicted and groundtruth pitch and yam')
    parser.add_argument('--norm_list', default='./data/norm_list.txt', help='path to norm list')
    parser.add_argument('--save_fig', default='./data/perdict_val_fig/', help='optional path for save fig')
    parser.add_argument('--lines_file', default='./sreen_camera_cail_imgs/1_upperleft/lines.txt', help='optional path for saved lines file')
    parser.add_argument('--cailbrated_file', default='./cailbrate_images/drawchessboard/', help='optional path for camera matrix and distortion file')

    args=parser.parse_args()
    # logging.debug(args)
    return args

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

def show__hp_distribution(args):

    with open(args.norm_list, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        num = len(lines)
        head_pose = np.zeros(shape=(num, 2))
        # head_list = []
        for i in range(num):
            line = lines[i].strip().split()
            head_pose[i] = np.asarray(line[4:6], dtype=np.float32)
        print(head_pose.shape)
        print(head_pose[0])
        drawScatter(head_pose[:, 0],head_pose[:, 1],'head pose')
        plt.show()

# 绘制散点图
def drawScatter(x,y, xlabel,ylabel, title=None):
    # 创建散点图
    # 第一个参数为点的横坐标
    # 第二个参数为点的纵坐标
    font2 = {
    'size'   : 18,
    }
    #设置坐标刻度值的大小以及刻度值的字体
    plt.tick_params(labelsize=18)
    plt.scatter(x,y, linewidths=0.01)
    plt.xlabel(xlabel, font2)
    plt.ylabel(ylabel,font2)


    # labels = get_xticklabels() + get_yticklabels()
    # [label.set_fontname('Times New Roman') for label in labels]

    # plt.title(title)
    # plt.show()

def dataDistribution(args):
    '''show and save gt and predict data distribution '''
    # for person in  os.listdir(args.gaze_data):
    gaze_data = np.load(args.gaze_data, allow_pickle=True)

    #drawScatter(gaze_data[:,0], gaze_data[:, 1], '{}-gt-pred'.format(person))
    #drawScatter(gaze_data[:,2], gaze_data[:, 3], '{}-gt-pred'.format(person))
    #plt.savefig('{}{}-yam-gt-pred.png'.format(args.save_fig, person))
    #plt.figure()
    plt.figure()
    drawScatter(gaze_data[:,0], gaze_data[:, 2], 'x','y')
    plt.plot([-0.5,0.5],[-0.5,0.5], color='r', alpha=0.8, label='y=x')
    plt.savefig('{}-pitch-gt-pred.png'.format(args.save_fig))

    plt.figure()
    drawScatter(gaze_data[:,1], gaze_data[:, 3],'x','y')
    plt.plot([-0.5,0.5],[-0.5,0.5], color='r', alpha=0.8, label='y=x')
    plt.savefig('{}-yaw-gt-pred.png'.format(args.save_fig))
        # plt.show()


        

def dataFit(args):
    '''fit from predict to gt'''
    save_dict = {}
    mean_error = []
    for person in  os.listdir(args.gaze_data):
        gaze_data = np.load(args.gaze_data+person, allow_pickle=True)
        error = []
        cail_num = 64
        val_num = 500
    
        nums = gaze_data.shape[0]
        # random sampling numbers
        for i in range(10):
            total_sample = list(range(0 , nums-val_num, 1))
            samples_index = random.sample(total_sample, cail_num)
            samples = np.zeros(shape=(cail_num, 9))
            for j, index in enumerate(samples_index):
                samples[j] = gaze_data[index]
            
            # create lr model
            # lr = linear_model.LinearRegression(normalize=False)
            # lr.fit(samples[:, 2:4], samples[:, 0:2])
            # y = lr.predict(gaze_data[value[1]-val_num : value[1], 2:4])

            lr = linear_model.LinearRegression(normalize=False)
            sample_weights = [1*(1-abs(x)) for x in samples[:, 2]]
            lr.fit(samples[:, 2].reshape(-1, 1), samples[:, 0].reshape(-1, 1), sample_weight=sample_weights)
            y_pitch = lr.predict(gaze_data[nums-val_num : , 2].reshape(-1, 1))

            lr = linear_model.LinearRegression(normalize=False)
            sample_weights = [1*(1-abs(x)) for x in samples[:, 3]]
            lr.fit(samples[:, 3].reshape(-1, 1), samples[:, 1].reshape(-1, 1), sample_weight=sample_weights)
            y_yaw = lr.predict(gaze_data[nums-val_num : , 3].reshape(-1, 1))

            y = np.concatenate((y_pitch, y_yaw), axis=1)

            mean = mean_angle_error(y, gaze_data[nums-val_num : , 0:2],
                                    gaze_data[nums-val_num : , 6:9])
            # print(mean)
            error.append(mean)

            # plot 2D data distribution
            # drawScatter(gaze_data[value[1]-val_num : value[1], 0], 
            #             gaze_data[value[1]-val_num : value[1], 1], '{}'.format(keys))
            # drawScatter(y[:, 0], y[:, 1], '{} linear fit'.format(keys))
            # plot 1D data distribution
            # drawScatter(gaze_data[value[1]-val_num : value[1], 0], 
            #             gaze_data[value[1]-val_num : value[1], 0], '{}'.format(keys))
            # drawScatter( gaze_data[value[1]-val_num : value[1], 0], y[:, 0], '{} linear fit'.format(keys))
            # # # plt.savefig('{}{}_linear_fit.png'.format(args.save_fig, keys))
            # plt.show()
        # print('lr w: {} intercept_: {}'.format(lr.coef_, lr.intercept_))
        print('{} num: {} mean: {}'.format(person,nums, np.mean(error)) )  
        mean_error.append(np.mean(error)) 
        save_dict[person] = np.mean(error)

    print('all person mean: ', np.mean(mean_error)) 
    save_dict['all-person'] = np.mean(mean_error) 

    with open(args.save_fig+'cail_error.json','w') as json_file:
        json.dump(save_dict, json_file, ensure_ascii=False)
     

def gazeErrorAndHeadPose(args):
     for person in  os.listdir(args.gaze_data):
        gaze_data = np.load(args.gaze_data+person, allow_pickle=True)

        # for keys, value in mpi_index.items():
        #     pitch_error = gaze_data[:, 0] - gaze_data[:, 2]
        #     yaw_error = gaze_data[:, 1] - gaze_data[:, 3]
        # # 绘制散点图

        pitch_error = gaze_data[:, 0] - gaze_data[:, 2]
        yaw_error = gaze_data[:, 1] - gaze_data[:, 3]

        print('pitch mean error: ', np.mean(abs(pitch_error)))
        print('yaw mean error: ', np.mean(abs(yaw_error)))

        print('pitch std error: ', np.std(abs(pitch_error)))
        print('yaw std error: ', np.std(abs(yaw_error)))

        # drawScatter(gaze_data[:,0], pitch_error, 'pitch-error') 
        # drawScatter(gaze_data[:,1], yaw_error, 'pitch-yaw-error')
        # plt.savefig('{}pitchyam-error.png'.format(args.save_fig))

        # fig = plt.figure()
        # ax = Axes3D(fig)
        # ax.scatter(gaze_data[:, 4], gaze_data[:, 5], pitch_error)
        # plt.xlabel('x')
        # plt.ylabel('y')
        # plt.ylabel('z')
        # ax.scatter(gaze_data[:, 4], gaze_data[:, 5], yaw_error)

        drawScatter(gaze_data[:,4], pitch_error, 'pitch-error') 
        drawScatter(gaze_data[:,5], yaw_error, 'headpitch-pitch-yaw-error')
        # plt.savefig('{}headyaw-pitch-yaw-error.png'.format(args.save_fig))

        plt.show()


if __name__ == '__main__':
    args = parse_args()

    dataDistribution(args)

    # dataFit(args)

    # show__hp_distribution(args)

    # gazeErrorAndHeadPose(args)
    