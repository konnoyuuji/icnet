#!/usr/bin/python3

import argparse
import time
import json

import numpy as np
from keras import backend as K
from keras.utils import multi_gpu_model
from keras.models import load_model
import tensorflow as tf
import cv2
import model
import glob
import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from utils import apply_color_map
np.set_printoptions(threshold=np.inf)

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', type=str, default=None, help='path to model checkpoint')
parser.add_argument('--test_image', type=str, default='output/input_sample.jpg', help='path to input test image')
opt = parser.parse_args()

print(opt)

#### Test ####
# Workaround to forbid tensorflow from crashing the gpu
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)

# Model
if opt.checkpoint:
    model = model.build_bn(640,320, 256, train=True) #DBG
    net = model.load_weights(opt.checkpoint)
else:
    print('No checkpoint specified! Set it with the --checkpoint argument option')
    exit()

indir = opt.test_image + "/*.jpg"
image_list = sorted(glob.glob(indir))
print(image_list)

for img_path in image_list:
    img_path = img_path.strip('\n')

    print(img_path)
    img = cv2.imread(img_path)
    image_width = img.shape[1]
    image_height = img.shape[0]

    # Testing
    x = np.array([cv2.resize(cv2.imread(img_path, 1), (640,320))])

    start_time = time.time()
    y = model.predict(np.array(x), batch_size=12)
    duration = time.time() - start_time

    print('Generated segmentations in %s seconds -- %s FPS' % (duration, 1.0/duration))

    # Save output image
    result = np.argmax(np.squeeze(y[0]), axis=-1).astype(np.uint8)
    result_img = Image.fromarray(result, mode='P')
    result_img = result_img.resize((image_width, image_height)) #DBG

    basename = os.path.basename(img_path)
    file, ext = os.path.splitext(basename)
    print(file)
    
    out = 'output/' + file + '.png'
    result_img.save(out)
