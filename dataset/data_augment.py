import cv2
import numpy as np
import random


def _mirror(image, gaze, head):
    if random.randrange(2):
        image = image[:, ::-1]
        gaze[1] = -gaze[1]
        head[1] = -head[1]

    return image, gaze, head


class preproc(object):

    def __init__(self):
        

    def __call__(self, image, gaze, head):

        return _mirror(image, gaze, head)
