#!/usr/bin/python3

import argparse

from keras import optimizers
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from keras.callbacks import LearningRateScheduler
from keras.utils import multi_gpu_model
from keras.models import load_model
from keras.models import model_from_json
from keras import backend as K
import tensorflow as tf

from utils import PolyDecay
from utils import MapillaryGenerator
import model

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=16, help='input batch size')
parser.add_argument('--image_width', type=int, default=640, help='the input image width')
parser.add_argument('--image_height', type=int, default=320, help='the input image height')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate of the optimizer')
parser.add_argument('--decay', type=float, default=0.9, help='learning rate decay (per epoch)')
parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--n_gpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--n_cpu', type=int, default=8, help='number of CPU threads to use during data generation')
parser.add_argument('--checkpoint', type=str, default=None, help='path to model checkpoint to resume training')
parser.add_argument('--epoch', type=int, default=0, help='epoch to start training')
opt = parser.parse_args()

print(opt)

#### Train ####

# Workaround to forbid tensorflow from crashing the gpu
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)

# Callbacks
checkpoint = ModelCheckpoint('output/weights.{epoch:03d}-{val_conv6_cls_categorical_accuracy:.3f}.h5', monitor='val_conv6_cls_categorical_accuracy', mode='max')
tensorboard = TensorBoard(batch_size=opt.batch_size)
lr_decay = LearningRateScheduler(PolyDecay(opt.lr, opt.decay, opt.n_epochs).scheduler)

# Generators
train_generator = MapillaryGenerator(batch_size=opt.batch_size, crop_shape=(opt.image_width, opt.image_height))
val_generator = MapillaryGenerator(mode='validation', batch_size=opt.batch_size, crop_shape=None, resize_shape=(opt.image_width, opt.image_height))

# Optimizer
optim = optimizers.SGD(lr=opt.lr, momentum=0.9)

# Model
net = model.build_bn(opt.image_width, opt.image_height, 256, train=True) 

# Training
net.compile(optim, 'categorical_crossentropy', loss_weights=[1.0, 0.4, 0.16], metrics=['categorical_accuracy'])
net.fit_generator(train_generator, len(train_generator), opt.n_epochs, callbacks=[checkpoint, tensorboard, lr_decay], 
                    validation_data=val_generator, validation_steps=len(val_generator), workers=opt.n_cpu, 
                    use_multiprocessing=False, shuffle=True, max_queue_size=10, initial_epoch=opt.epoch)                   
###############
