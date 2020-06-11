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

from dataset.datasets import MPIIDatasets
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


def train(args, train_loader, model, auxiliarynet, criterion, optimizer,
          epoch):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    error = AverageMeter('error', ':6.2f')

    progress = ProgressMeter(len(train_loader), batch_time, data_time, losses, error,
                             prefix="Train Epoch: [{}]".format(epoch))
    # switch to train mode
    model.train()
    auxiliarynet.train()
    end = time.time()
    for batch_idx, (patch, gaze_norm_g, head_norm, rot_vec_norm) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        patch.requires_grad = False
        patch = patch.to(args.device)

        gaze_norm_g.requires_grad = False
        gaze_norm_g = gaze_norm_g.to(args.device)

        head_norm.requires_grad = False
        head_norm = head_norm.to(args.device)

        rot_vec_norm.requires_grad = False
        rot_vec_norm = rot_vec_norm.to(args.device)

        # model = model.to(args.device)
        # auxiliarynet = auxiliarynet.to(args.device)
        gaze_pred, features = model(patch)
        # print(features.size())
        hp_pred = auxiliarynet(features)
        head_norm = 100 * head_norm
        gaze_norm_g = 100 * gaze_norm_g
        loss = criterion(gaze_norm_g, head_norm, gaze_pred, hp_pred)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        angle_error = mean_angle_error(gaze_pred.cpu().detach().numpy()/100,
                                       gaze_norm_g.cpu().detach().numpy()/100, 
                                       rot_vec_norm.cpu().detach().numpy())
        
        losses.update(loss.item())
        error.update(angle_error)

        if(batch_idx + 1) % args.print_freq == 0: 
            progress.print(batch_idx+1)
    return losses.get_avg(), error.get_avg()


def validate(args, val_dataloader, model, auxiliarynet, criterion,
             epoch):
    
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    error = AverageMeter('error', ':6.2f')

    progress = ProgressMeter(len(val_dataloader), batch_time, data_time, losses, error,
                             prefix="Val Epoch: [{}]".format(epoch))

    model.eval()
    auxiliarynet.eval()
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
            gaze_pred, features = model(patch)
            hp_pred = auxiliarynet(features)
            head_norm = 100 * head_norm
            gaze_norm_g = 100 * gaze_norm_g
            loss = criterion(gaze_norm_g, head_norm, gaze_pred, hp_pred)

            angle_error= mean_angle_error(gaze_pred.cpu().detach().numpy()/100, 
                                          gaze_norm_g.cpu().detach().numpy()/100,
                                          rot_vec_norm.cpu().detach().numpy())
            losses.update(loss.item())
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

    if args.pretrained:
        model = EfficientNet.from_pretrained(args.arch).to(args.device)
        print("=> using pre-trained model '{}'".format(args.arch))
    else:
        print("=> creating model '{}'".format(args.arch))
        model = EfficientNet.from_name(args.arch).to(args.device)
    auxiliarynet = AuxiliaryNet().to(args.device)
    criterion = GazeLoss()
    optimizer = torch.optim.Adam(
        [{
            'params': model.parameters()
        }],
        lr=args.base_lr,
        weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=args.lr_patience, verbose=True, min_lr=args.min_lr)

 # optionally resume from a checkpoint
    min_error = 1e6
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            min_error = checkpoint['error']
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {}) {:.3f}"
                  .format(args.resume, checkpoint['epoch'], min_error))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # step 3: data
    # argumetion
    transform = transforms.Compose([transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])])

    mpiidataset = MPIIDatasets(args.dataroot, train=True, transforms=transform)
    train_dataloader = DataLoader(
        mpiidataset,
        batch_size=args.train_batchsize,
        shuffle=True,
        num_workers=args.workers,
        drop_last=False)

    mpii_val_dataset = MPIIDatasets(args.val_dataroot, train=False, transforms=transform)
    val_dataloader = DataLoader(
        mpii_val_dataset,
        batch_size=args.val_batchsize,
        shuffle=False,
        num_workers=args.workers)
       


    # step 4: run
    writer = SummaryWriter(args.tensorboard)
    for epoch in range(args.start_epoch, args.end_epoch + 1):
        train_loss, train_error = train(args, train_dataloader, model,
                            auxiliarynet, criterion, optimizer, epoch)

        val_loss, val_error = validate(args, val_dataloader, model,
                            auxiliarynet, criterion, epoch)
        filename = os.path.join(
            str(args.snapshot), "checkpoint_epoch_" + str(epoch) + '.pth.tar')
        is_best = min_error > val_error
        min_error = min(min_error, val_error)
        save_checkpoint({
            'epoch': epoch+1,
            'model': model.state_dict(),
            'optimizer' : optimizer.state_dict(),
            'error': min_error,
        }, is_best, filename)
        scheduler.step(val_loss)
        writer.add_scalars('data/error', {'val error': val_error, 'train error ': train_error}, epoch)
        writer.add_scalars('data/loss', {'val loss': val_loss, 'train loss': train_loss}, epoch)
    writer.close()


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
    parser.add_argument('--base_lr', default=0.0001, type=float)
    parser.add_argument('--weight-decay', '--wd', default=1e-5, type=float)
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')

    # -- lr
    parser.add_argument("--lr_patience", default=10, type=int)
    parser.add_argument("--min_lr", default=1e-6, type=float)

    # -- epoch
    parser.add_argument('--start_epoch', default=1, type=int)
    parser.add_argument('--end_epoch', default=1000, type=int)

    # -- net arch
    parser.add_argument('-a', '--arch', metavar='ARCH', default='efficientnet-b0',
                    help='model architecture (default: efficientnet-b0)')
    parser.add_argument('--pretrained', default = True, dest='pretrained', action='store_true',
                    help='use pre-trained model')    

    # -- snapshot、tensorboard log and checkpoint
    parser.add_argument(
        '--snapshot',
        default='./checkpoint/snapshot/',
        type=str,
        metavar='PATH')
    parser.add_argument(
        '--log_file', default="./checkpoint/train.logs", type=str)
    parser.add_argument(
        '--tensorboard', default="./checkpoint/tensorboard", type=str)
    parser.add_argument('--resume', default='model_best.pth.tar', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

    # --dataset
    parser.add_argument(
        '--dataroot',
        default='./data/MPIIFaceGaze/norm_list.txt',
        type=str,
        metavar='PATH')
    parser.add_argument(
        '--val_dataroot',
        default= './data/MPIIFaceGaze/norm_list.txt',
        type=str,
        metavar='PATH')
    parser.add_argument('--train_batchsize', default=16, type=int)
    parser.add_argument('--val_batchsize', default=4, type=int)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
