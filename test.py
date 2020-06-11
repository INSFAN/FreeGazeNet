#-*- coding:utf-8 -*-

import argparse
import logging
from pathlib import Path
import time
import os
import shutil

import numpy as np
import torch
import cv2 as cv

from torch.utils import data
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
from torch.autograd import Variable

from dataset.datasets import MPIIDatasets, GazeCaptureDatasets
from models.gaze_mobilenetv3 import MobileNetV3
from models.efficientnet import EfficientNet, AuxiliaryNet
from loss.loss import GazeLoss, L2Loss, L1Loss
from utils.utils import AverageMeter, ProgressMeter, mean_angle_error


def print_args(args):
    for arg in vars(args):
        s = arg + ': ' + str(getattr(args, arg))
        logging.info(s)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    if is_best:
        torch.save(state, filename)
        logging.info('Save checkpoint to {0:}'.format(filename))
        shutil.copyfile(filename, 'model_best.pth.tar')


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected')


def validate(args, val_dataloader, model, auxiliarynet, epoch):
    
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    error = AverageMeter('error', ':6.2f')

    progress = ProgressMeter(len(val_dataloader), batch_time, data_time, losses, error,
                             prefix="Val Epoch: [{}]".format(epoch))

    model.eval()
    # auxiliarynet.eval()
    end = time.time()
    with torch.no_grad():
        end = time.time()
        for i, (patch, gaze_norm_g, head_norm, rot_vec_norm) in enumerate(val_dataloader):
            # measure data loading time
            data_time.update(time.time() - end)
            patch = patch.to(args.device)
            gaze_norm_g = gaze_norm_g.to(args.device)

            head_norm = head_norm.to(args.device)

            rot_vec_norm = rot_vec_norm.to(args.device)

            # model = model.to(args.device)
            gaze_pred, _ = model(patch)
            # hp_pred = auxiliarynet(features)
            

            head_norm = 10 * head_norm
            gaze_norm_g = 10 * gaze_norm_g
            # loss = criterion(gaze_norm_g, head_norm, gaze_pred[:,0:2], gaze_pred[:,2:4])

            angle_error = mean_angle_error(gaze_pred.cpu().detach().numpy()/10, 
                                          gaze_norm_g.cpu().detach().numpy()/10,
                                          rot_vec_norm.cpu().detach().numpy())
            # losses.update(loss.item())
            error.update(angle_error)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()            

            if (i+1) % args.print_freq == 0:
                progress.print(i+1)
                # img = patch.cpu().detach().numpy()[0].deepcopy()
                # to_visualize = draw_gaze(img[0], (0.25 * img.shape[1], 0.25 * img.shape[1]), gaze_pred, 
                # gaze_norm_g, length=80.0, thickness=1)
                # cv2.imshow('vis', to_visualize)
                # cv2.waitKey(1)

    return losses.get_avg(), error.get_avg()


def main(args):
    # Step 1: parse args config
    logging.basicConfig(
        format=
        '[%(asctime)s] [p%(process)s] [%(pathname)s:%(lineno)d] [%(levelname)s] %(message)s',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(args.log_file, mode='w'),
            logging.StreamHandler()
        ])
    print_args(args)

    # Step 2: model, criterion, optimizer, scheduler
    # model = MobileNetV3(mode='large').to(args.device)


    model = EfficientNet.from_name(args.arch).to(args.device)
    # auxiliarynet = AuxiliaryNet().to(args.device)
    auxiliarynet = None

    checkpoint = torch.load(args.model_path)
    model.load_state_dict(checkpoint['model'])


    # step 3: data
    # argumetion
    transform = transforms.Compose([transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])])

    # mpiidataset = MPIIDatasets(args.dataroot, train=True, transforms=transform)
    # train_dataset = GazeCaptureDatasets(args.dataroot, train=True, transforms=transform)

    # mpii_val_dataset = MPIIDatasets(args.val_dataroot, train=False, transforms=transform)
    val_dataset = GazeCaptureDatasets(args.val_dataroot, train=True, transforms=transform)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.val_batchsize,
        shuffle=False,
        num_workers=args.workers)
       

    # step 4: run
    val_loss, val_error = validate(args, val_dataloader, model,
                        auxiliarynet, 1)
    print("val_loss: '{}' val_error: '{}'".format(val_loss, val_error))


def parse_args():
    parser = argparse.ArgumentParser(description='Trainging Template')
    # general
    parser.add_argument('-j', '--workers', default=0, type=int)
    parser.add_argument('--devices_id', default='0', type=str)  #TBD
    parser.add_argument('--device', default=torch.device("cuda" if torch.cuda.is_available() else "cpu"), type=str)
    parser.add_argument('--test_initial', default='false', type=str2bool)  #TBD
    parser.add_argument('-p', '--print-freq', default=100, type=int,
                        metavar='N', help='print frequency (default: 10)')
    # training
    ##  -- optimizer
    parser.add_argument('--base_lr', default=0.001, type=float)
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float)
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')


    # -- net arch
    parser.add_argument('-a', '--arch', metavar='ARCH', default='efficientnet-b0',
                    help='model architecture (default: efficientnet-b0)')   


    parser.add_argument(
        '--log_file', default="./checkpoint/train.logs", type=str)
    parser.add_argument(
        '--tensorboard', default="./checkpoint/tensorboard", type=str)
    parser.add_argument('--model_path', default='model_best.pth.tar', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

    # --dataset
    # parser.add_argument(
    #     '--dataroot',
    #     default='./data/MPIIFaceGaze/norm_list.txt',
    #     type=str,
    #     metavar='PATH')
    parser.add_argument(
        '--val_dataroot',
        default= './data/MPIIFaceGaze/norm_list.txt',
        type=str,
        metavar='PATH')
    # parser.add_argument(
    #     '--dataroot',
    #     default='./data/GazeCapture/norm_list.txt',
    #     type=str,
    #     metavar='PATH')
    # parser.add_argument(
    #     '--val_dataroot',
    #     default= './data/GazeCapture/norm_list.txt',
    #     type=str,
    #     metavar='PATH')
    parser.add_argument('--val_batchsize', default=16, type=int)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
