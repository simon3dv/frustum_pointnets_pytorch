# frustum_pointnets_pytorch
A pytorch version of [frustum-pointnets](https://github.com/charlesq34/frustum-pointnets) 
(Not support Pointnet++ yet)

## Keep updating, README is false,**don't use it right now**!
## Requirements
Test on 
* Ubuntu-18.04
* CUDA-10.0
* Pytorch 1.3
* python 3.7


## Usage
### Installation
Some visulization demos need mayavi,I install mayavi(python3) by:
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
CUDA_VISIBLE_DEVICES=0 python train/train_fpointnets.py
```
### nuScenes2Kitti
```angular2
CUDA_VISIBLE_DEVICES=0 python train/train_fpointnets.py --dataset nuscenes2kitti
```

## Test
### Kitti
```angular2
CUDA_VISIBLE_DEVICES=0 python train/test_fpointnets.py
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
```
test segmentation accuracy: 0.902415
test box IoU(ground/3D): 0.800471/0.748862
test box estimation accuracy (IoU=0.7): 0.776121
(batch_scores = mask_mean_prob)
car  AP @0.70, 0.70,  0.70:
bbox AP:100.00,100.00,100.00
bev  AP:85.76, 80.21, 74.20
3d   AP:68.41, 63.89, 66.46
```



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

