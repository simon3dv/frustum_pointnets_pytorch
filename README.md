# frustum_pointnets_pytorch
A pytorch version of [frustum-pointnets](https://github.com/charlesq34/frustum-pointnets) 

## Requirements
Test on 
* Ubuntu-18.04
* CUDA-10.0
* Pytorch 1.3
* python 3.7


## Usage

### Prepare Training Data
```angular2
frustum_pointnets_pytorch
├── datase
│   ├── KITTI
│   │   ├── ImageSets
│   │   ├── object
│   │   │   ├──training
│   │   │      ├──calib & velodyne & label_2 & image_2 & (optional: planes)
│   │   │   ├──testing
│   │   │      ├──calib & velodyne & image_2
│   ├── nuScenes
│   │   ├── v1.0-mini
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
To visulize Kitti
```angular2
python nuscenes2kitti/prepare_data.py --demo
```
To prepare training data
```angular2
python kitti/prepare_data.py --gen_train --gen_val --gen_val_rgb_detection --car_only
```

#### nuScenes
TODO

#### nuScenes2Kitti
First, convert nuScenes to kitti(only consider CAM_FRONT)
```angular2
python nuscenes2kitti/nuscenes_convert_full_sample.py --version 'v1.0-mini' --CAM_FRONT_only
python nuscenes2kitti/nuscenes_convert_full_sample.py --version 'v1.0-trainval' --CAM_FRONT_only --number 7481
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
CUDA_VISIBLE_DEVICES=0 python train/train.py
```
### nuScenes2Kitti
```angular2
CUDA_VISIBLE_DEVICES=0 python train/train.py --dataset nuscenes2kitti
```

## Test
### Kitti
```angular2
CUDA_VISIBLE_DEVICES=0 python train/test.py
train/kitti_eval/evaluate_object_3d_offline dataset/KITTI/object/training/label_2/ train/test_results
```
### nuScenes2Kitti
```angular2

```

## Results
```angular2
CUDA_VISIBLE_DEVICES=2 python train/train.py --name 20200121-decay_rate=0.7-decay_step=20_caronly --decay_rate 0.7 --decay_step 20 --datatype caronly
**** EPOCH 150 ****
Epoch 150/150:
100%|███████████████████████████████████████████████████████| 1851/1851 [04:04<00:00,  7.57it/s]
[150: 1850/1851] train loss: 0.052071
segmentation accuracy: 0.961067
box IoU(ground/3D): 0.847253/0.792872
box estimation accuracy (IoU=0.7): 0.886701
100%|█████████████████████████████████████████████████████████| 392/392 [00:22<00:00, 17.70it/s]
[150] test loss: 0.103528
test segmentation accuracy: 0.902443
test box IoU(ground/3D): 0.800930/0.748648
test box estimation accuracy (IoU=0.7): 0.772930
learning rate: 0.000082
Best Test acc: 0.777317(Epoch 131)
Time 11.087114362829999 hours

CUDA_VISIBLE_DEVICES=0 python train/test.py --model_path log/20200121-decay_rate=0.7-decay_step=20_caronly/20200121-decay_rate=0.7-decay_step=20_caronly-acc0.777317-epoch130.pth --return_all_loss --output train/kitti_caronly_v1
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████| 392/392 [00:24<00:00, 16.04it/s]
[1] test loss: 0.103475
test segmentation accuracy: 0.902415
test box IoU(ground/3D): 0.800471/0.748862
test box estimation accuracy (IoU=0.7): 0.776121
```
## TODO List
* features
  - [ ] pointnet++
* models
  - [ ] frustum-convnet
* datasets
  - [ ] nuScenes
  - [ ] SUN-RGBD

  
# Acknowledgement
* [frustum-pointnets](https://github.com/charlesq34/frustum-pointnets) 
* [frustum-convnet](https://github.com/zhixinwang/frustum-convnet)
* [Pointnet_Pointnet2_pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch)
* [NuScenes2KITTI](https://github.com/zcc31415926/NuScenes2KITTI)
* [second.pytorch](https://github.com/traveller59/second.pytorch)
* [nuscenes-devkit](https://github.com/nutonomy/nuscenes-devkit)
* [Det3D](https://github.com/poodarchu/Det3D)

