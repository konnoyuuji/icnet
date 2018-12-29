# Keras ICNet 

## Dataset
- Camvid

## Train
```
python train.py --batch_size 4 --n_epochs 10000 --decay 0.0001 --image_width 800 --image_height 640
tensorboard --logdir=./logs
http://localhost:6006/
```

## Test
```
python test.py --checkpoint weights.3055-0.890.h5 --test_image datasets/mapillary/testing/images/
```
