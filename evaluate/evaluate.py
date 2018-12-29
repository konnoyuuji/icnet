import numpy as np
import matplotlib.pyplot as plt
from pylab import *
import os
import sys
import time
import cv2
from PIL import Image
from keras.preprocessing.image import *
from keras.models import load_model
import keras.backend as K

def calculate_iou(nb_classes, pred_dir, label_dir, image_list):
    conf_m = zeros((nb_classes, nb_classes), dtype=float)
    total = 0
    mean_acc = 0.
    for img_num in image_list:
        img_num = img_num.strip('\n')
        total += 1
        print('#%d: %s' % (total, img_num))
        pred = img_to_array(Image.open('%s/%s.png' % (pred_dir, img_num))).astype(int)
        label = img_to_array(Image.open('%s/%s.png' % (label_dir, img_num))).astype(int)
        flat_pred = np.ravel(pred)
        flat_label = np.ravel(label)
        for p, l in zip(flat_pred, flat_label):
            if l == 255:
                continue
            if l < nb_classes and p < nb_classes:
                conf_m[l, p] += 1
            else:
                print('Invalid entry encountered, skipping! Label: ', l,
                      ' Prediction: ', p, ' Img_num: ', img_num)
    I = np.diag(conf_m)
    U = np.sum(conf_m, axis=0) + np.sum(conf_m, axis=1) - I
    IOU_tmp = I/U
    IOU = zeros(11, dtype=float)
    category = []
    total = 0

    # Pickup CamVid dataset evaluate 11 category
    IOU[0]  = IOU_tmp[97]  # Bicyclist
    IOU[1]  = IOU_tmp[38]  # Building
    IOU[2]  = IOU_tmp[33]  # Car
    IOU[3]  = IOU_tmp[184] # ColumPole
    IOU[4]  = IOU_tmp[71]  # Fence
    IOU[5]  = IOU_tmp[56]  # Pedestrian
    IOU[6]  = IOU_tmp[90]  # Road
    IOU[7]  = IOU_tmp[21]  # Sidewalk
    IOU[8]  = IOU_tmp[147] # SignSymbol
    IOU[9]  = IOU_tmp[128] # Sky
    IOU[10] = IOU_tmp[113] # Tree

    category = [
    'Bicyclist',
    'Building',
    'Car',
    'ColumPole',
    'Fence',
    'Pedestrian',
    'Road',
    'Sidewalk',
    'SignSymbol',
    'Sky',
    'Tree']

    for i in range(11):
        print(category[i], IOU[i])

    meanIOU = np.nanmean(IOU)
    return conf_m, IOU, meanIOU

def evaluate(nb_classes, val_file_path, label_dir):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    pred_dir = "./pred_dir"
    fp = open(val_file_path)
    image_list = fp.readlines()
    fp.close()

    conf_m, IOU, meanIOU = calculate_iou(nb_classes, pred_dir, label_dir, image_list)

    print('IOU: ')
    print(IOU)
    print('meanIOU: %f' % meanIOU)
    print('pixel acc: %f' % (np.sum(np.diag(conf_m))/np.sum(conf_m)))

if __name__ == '__main__':
    nb_classes = 256
    test_file_path = './test.txt'
    label_dir      = './label_dir'
    evaluate(nb_classes, test_file_path, label_dir)
