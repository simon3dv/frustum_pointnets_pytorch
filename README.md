# frustum_pointnets_pytorch
A pytorch version of [frustum-pointnets](https://github.com/charlesq34/frustum-pointnets) 
(Not support Pointnet++ yet)

main function of f-pointnets now:
train/train_fpointnets.py, 
train/test_fpointnets.py,
train/provider_fpointnet.py,
models/frustum_pointnets_v1_old.py
model_util_old.py
kitti/*

## Requirements
Test on 
* Ubuntu-18.04
* CUDA-10.0
* Pytorch 1.3
* python 3.7


## Usage
### Installation(optional)
Some visulization demos need mayavi, it would be a little bit difficult to install it.
I install mayavi(python3) by:
(ref:http://docs.enthought.com/mayavi/mayavi/installation.html#installing-with-pip) 
```angular2
1. download vtk(https://vtk.org/download/) and compile:
unzip VTK
cd VTK
mkdir build
cd build
cmake ..
make
sudo make install 
2. install mayavi and PyQt5
pip install mayavi
pip install PyQt5
```
### Prepare Training Data
```angular2
frustum_pointnets_pytorch
├── dataset
│   ├── KITTI
│   │   ├── ImageSets
│   │   ├── object
│   │   │   ├──training
│   │   │      ├──calib & velodyne & label_2 & image_2 & (optional: planes)
│   │   │   ├──testing
│   │   │      ├──calib & velodyne & image_2
│   ├── nuScenes
│   │   ├── v1.0-mini
│   │   │      ├──maps & samples & sweeps & v1.0-mini
│   │   ├── v1.0-trainval
│   │   ├── v1.0-test
│   ├── nuScenes2kitti
│   │   ├── v1.0-mini
│   │   ├── training
│   │   ├── testing
├── kitti
│   │   ├── image_sets
│   │   ├── rgb_detections
├── nuscenes2kitti
│   │   ├── image_sets
│   │   ├── rgb_detections(TODO)
├── train
```
#### Kitti
To visulize single sample in Kitti
```angular2
python kitti/prepare_data.py --demo
```
To prepare training data
```angular2
python kitti/prepare_data.py --gen_train --gen_val --gen_val_rgb_detection --car_only
```
To visulize all gt boxes and prediction boxes:
```angular2
python kitti/kitti_object.py
```

#### nuScenes
TODO

#### nuScenes2Kitti
First, convert nuScenes to kitti(only consider CAM_FRONT)
```angular2
python nuscenes2kitti/nuscenes_convert_full_samples.py --version v1.0-mini --CAM_FRONT_only
python nuscenes2kitti/nuscenes_convert_full_samples.py --version v1.0-trainval --CAM_FRONT_only --number 7481
```
After conversion, you can visulize them by 
```angular2
python nuscenes2kitti/prepare_data.py --demo
```
Then, generate *.pickle training data and write to nuscenes2kitti by 
```angular2
python nuscenes2kitti/prepare_data.py --gen_train --gen_val --gen_mini --car_only --CAM_FRONT_only
```

## train
### Kitti
```angular2
CUDA_VISIBLE_DEVICES=0 python train/train_fpointnets.py --log_dir log
```
### nuScenes2Kitti
```angular2
CUDA_VISIBLE_DEVICES=0 python train/train_fpointnets.py --dataset nuscenes2kitti
```

## Test
### Kitti
```angular2
CUDA_VISIBLE_DEVICES=0 python train/test_fpointnets.py --model_path <log/.../xx.pth> --output test_results
train/kitti_eval/evaluate_object_3d_offline dataset/KITTI/object/training/label_2/ test_results
```

### Visulize
```
python kitti/kitti_object.py
```


### nuScenes2Kitti
```angular2

```

## Results
### FrustumPointnetv1 from rgb detection on KITTI(val) Dataset
```angular2
(score_list.append(batch_rgb_prob[j])))
car  AP @0.70, 0.70,  0.70:
bbox AP:96.48, 90.31, 87.63
bev  AP:88.57, 84.78, 76.79
3d   AP:85.09, 72.11, 64.25
```

### FrustumPointnetv1 with 2D gt box on KITTI(val) Dataset
(using f-pointnets scores(max positive score of segmentation mask here, check `test.py`))
```
test segmentation accuracy: 0.901991
test box IoU(ground/3D): 0.796272/0.744235
test box estimation accuracy (IoU=0.7): 0.766470
(batch_scores = mask_max_prob)
car  AP @0.70, 0.70,  0.70:
bbox AP:100.00,100.00,100.00
bev  AP:87.66, 85.59, 77.76
3d   AP:84.79, 74.12, 66.64
```
(test from gt leads to 2~3 AP decrese, because f-pointnets cannot output a good "score")

### Model Time cost:
train per batch: 6.00ms/32 objects

test per batch: 15.21ms/32 objects

test per object: 135.83ms/object

## More Results(2020.3.7)
### Train from 2d ground truth
```python train/train_fpointnets.py --name three```
```
**** EPOCH 150 ****
Epoch 150/150:
100%|███████████████████████████████████████████████████| 1851/1851 [05:08<00:00,  6.00it/s]
[150: 1850/1851] train loss: 0.051460
segmentation accuracy: 0.960588
box IoU(ground/3D): 0.845676/0.790897
box estimation accuracy (IoU=0.7): 0.880115
100%|█████████████████████████████████████████████████████| 392/392 [00:25<00:00, 15.21it/s]
[150] test loss: 0.106460
test segmentation accuracy: 0.901991
test box IoU(ground/3D): 0.796272/0.744235
test box estimation accuracy (IoU=0.7): 0.766470
learning rate: 0.000082
Best Test acc: 0.769261(Epoch 130)
Time 14.12774204615861 hours
model saved to log/three_caronly_kitti_2020-03-05-16/three_caronly_kitti_2020-03-05-16/acc0.769261-epoch129.pth
```


#### Test with 2d ground truth
```
python train/test_fpointnets.py --output output/default --model_path log/three_caronly_kitti_2020-03-05-16/three_caronly_kitti_2020-03-05-16/acc0.769261-epoch129.pth
train/kitti_eval/evaluate_object_3d_offline dataset/KITTI/object/training/label_2/ output/default/
```

```
car_detection AP: 100.000000 100.000000 100.000000
car_detection_ground AP: 87.663536 85.591621 77.763123
car_detection_3d AP: 84.790321 74.122635 66.642036
```
![car_detection_3d_from_gt](https://github.com/simon3dv/frustum_pointnets_pytorch/blob/master/doc/results/car_detection_3d_fromgt.png)

#### Test with rgb detection results
```
python train/test_fpointnets.py --output output/gt2rgb --model_path log/three_caronly_kitti_2020-03-05-16/three_caronly_kitti_2020-03-05-16/acc0.769261-epoch129.pth --data_path kitti/frustum_caronly_val_rgb_detection.pickle --from_rgb_det
train/kitti_eval/evaluate_object_3d_offline dataset/KITTI/object/training/label_2/ output/gt2rgb/
```

using f-pointnets scores(use max positive score of segmentation mask here, check `test.py`)
```
car_detection AP: 94.753860 86.358948 84.424759
car_detection_ground AP: 85.699196 80.538277 73.625732
car_detection_3d AP: 81.882027 70.021523 63.075848
```
![car_detection_fromrgb_maxscore](https://github.com/simon3dv/frustum_pointnets_pytorch/blob/master/doc/results/car_detection_3d_fromrgb_maxscore.png)

using rgb detection scores
```
car_detection AP: 96.482544 90.305161 87.626389
car_detection_ground AP: 87.814728 82.850800 75.781403
car_detection_3d AP: 84.952042 72.118301 64.253830
```
![car_detection_3d_fromrgb_rgbscore](https://github.com/simon3dv/frustum_pointnets_pytorch/blob/master/doc/results/car_detection_3d_fromrgb_rgbscore.png)




## TODO List
* features
  - [ ] pointnet++
* models
  - [ ] frustum-convnet
  - [ ] extend_rgb,globalfusion,densefusion,pointpaint,...
* datasets
  - [ ] nuScenes2kitti
  - [ ] SUN-RGBD


  
# Acknowledgement
many codes are from:
* [frustum-pointnets](https://github.com/charlesq34/frustum-pointnets) 
* [frustum-convnet](https://github.com/zhixinwang/frustum-convnet)
* [Pointnet_Pointnet2_pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch)
* [NuScenes2KITTI](https://github.com/zcc31415926/NuScenes2KITTI)
* [second.pytorch](https://github.com/traveller59/second.pytorch)
* [nuscenes-devkit](https://github.com/nutonomy/nuscenes-devkit)
* [Det3D](https://github.com/poodarchu/Det3D)

