# Keras ICNet demo

## Train
```
python train.py --batch_size 1 --n_epochs 10000 --decay 0.0001
tensorboard --logdir=./logs
http://localhost:6006/
```
## Test
```
python test.py --checkpoint weights.h5 --test_image datasets/testing/images/
```
## Evaluate
```
cd evaluate
cp [label data] label_dir
cp [predict result data] pred_dir
ls label_dir|cut -d '.' -f 1|tee test.txt
python evaluate
```

You will see the result.
```
Bicyclist 0.0
Building 0.6568308122799638
Car 0.47078117857161916
ColumPole 0.0397699452834222
Fence 0.05348958626851695
Pedestrian 0.07584152620831534
Road 0.6953406930650506
Sidewalk 0.2250812830670305
SignSymbol 0.044979453954168455
Sky 0.8607907012051145
Tree 0.4562983175791105
IOU: 
[0.         0.65683081 0.47078118 0.03976995 0.05348959 0.07584153
 0.69534069 0.22508128 0.04497945 0.8607907  0.45629832]
meanIOU: 0.325382
pixel acc: 0.699343
``
