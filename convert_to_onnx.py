import argparse
import time

import numpy as np
import torch
import torchvision
from torchvision import transforms
import cv2
import onnx

# from dataset.datasets import MPIIDatasets, GazeCaptureDatasets
from dataset.datasets_union import MPIIDatasets, GazeCaptureDatasets
from models.gaze_mobilenetv3 import MobileNetV3
from models.efficientnet_union import EfficientNet, AuxiliaryNet
from loss.loss import GazeLoss, L2Loss, L1Loss, LogCosh
from utils.utils import AverageMeter, ProgressMeter, mean_angle_error


def main(args):
    print(onnx .__version__)
    print(torch.__version__)
    checkpoint = torch.load(args.model_path, map_location=args.device)
    model = EfficientNet.from_name(args.arch).to(args.device)
    auxiliarynet = MobileNetV3(mode='small').to(args.device)
    model.load_state_dict(checkpoint['model'])
    auxiliarynet.load_state_dict(checkpoint['aux'])
    model.eval()
    auxiliarynet.eval()

    ##################export###############
    output_onnx = 'gaze_mpii_face.onnx'
    print("==> Exporting model to ONNX format at '{}'".format(output_onnx))
    input_names = ["input"]
    output_names = ["face_feature"]
    inputs_face = torch.randn(1, 3, 128, 96).to(args.device)
    face_torch_out = torch.onnx._export(auxiliarynet, inputs_face, output_onnx, export_params=True, verbose=False,
                                   input_names=input_names, output_names=output_names, opset_version=9,
                                   keep_initializers_as_inputs=True)

    output_gaze_onnx = 'gaze_mpii_estimator.onnx'
    print("==> Exporting model to ONNX format at '{}'".format(output_gaze_onnx))
    gaze_input_names = ["input", "face_feature"]
    output_names = ["gaze"]
    inputs_eyes = torch.randn(1, 3, 320, 64).to(args.device)
    face_feature = torch.randn(1, 1152).to(args.device)
    gaze_torch_out = torch.onnx._export(model, (inputs_eyes, face_feature), output_gaze_onnx, export_params=True, verbose=False,
                                   input_names=gaze_input_names, output_names=output_names, opset_version=9,
                                   keep_initializers_as_inputs=True)
    ##################end##############

    ###################check##############
    print("==> check model to ONNX format at '{}'".format(output_onnx))
    model_onnx = onnx.load(output_onnx)
    onnx.checker.check_model(model_onnx)

    print("==> check model to ONNX format at '{}'".format(output_gaze_onnx))
    model_onnx = onnx.load(output_gaze_onnx)
    onnx.checker.check_model(model_onnx)
    #################end#######################
# python -m onnxsim faceDetector.onnx faceDetector_sim.onnx

def parse_args():
    parser = argparse.ArgumentParser(description='Testing')
    parser.add_argument('--device', default=torch.device("cuda" if torch.cuda.is_available() else "cpu"), type=str)
    parser.add_argument('-a', '--arch', metavar='ARCH', default='efficientnet-b0',
                    help='model architecture (default: efficientnet-b0)')
    parser.add_argument(
        '--model_path',
        default="./checkpoint/snapshot/mp14_4.786_model_best.pth.tar",
        type=str)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)