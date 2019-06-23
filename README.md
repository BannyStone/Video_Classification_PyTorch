# Video-Classification-Pytorch

***Still in development***.

This is a repository containing 3D models and 2D models for video classification. The code is based on PyTorch 1.0.
Until now, it supports the following datasets:
Kinetics-400, Mini-Kinetics-200, UCF101, HMDB51

## Results

### Kinetics-400

We report the baselines with ResNet-50 backbone on Kinetics-400 validation set as below (all models are trained on training set).
All the models are trained in one single server with 8 GTX 1080 Ti GPUs.

| <sub>network</sub> | <sub>pretrain data</sub> | <sub>spatial resolution</sub> | <sub>input frames</sub> | <sub>sampling stride</sub> | <sub>backbone</sub> | <sub>top1</sub> | <sub>top5</sub> |
| ------------------ | ------------------ | ------------------ | ------------------ | ------------------ | ------------------ | ------------------ | ------------------ |
| <sub>ResNet50-SlowOnly</sub> | <sub>ImageNet-1K</sub> | <sub>224x224</sub> | <sub>8</sub> | <sub>8</sub> | <sub>ResNet50</sub> | <sub>73.77</sub> | <sub>91.17</sub> |


## Get the Code
```Shell
git clone --recursive https://github.com/BannyStone/Video_Classification_PyTorch.git
```

## Preparing Dataset
### Kinetics-400
```Shell
cd data/kinetics400
mkdir access && cd access
ln -s $YOUR_KINETICS400_DATASET_TRAIN_DIR$ RGB_train
ln -s $YOUR_KINETICS400_DATASET_VAL_DIR$ RGB_val
```
Note that:
- The reported models are trained with the Kinetics data provided by Xiaolong Wang.https://github.com/facebookresearch/video-nonlocal-net/blob/master/DATASET.md
- In train and validation lists for all datasets, each line represents one video where the first element is the video frame directory, the second element is the number of frames and the third element is the index of class. Please prepare your own list accordingly because different video parsing method may lead to different frame numbers. We show part of Kinetics-400 train list as an example:
```shell
RGB_train/D32_1gwq35E 300 66
RGB_train/-G-5CJ0JkKY 250 254
RGB_train/4uZ27ivBl00 300 341
RGB_train/pZP-dHUuGiA 240 369
```
- This code can read the image files in each video frame folder according to the image template argument *image_tmpl*, such as *image_{:06d}.jpg*.

## Training
Execute training script:
```Shell
./scripts/kinetics400_3d_res50_slowonly_im_pre.sh
```

We show script *kinetics400_3d_res50_slowonly_im_pre.sh* here:
```Shell
python main.py \
kinetics400 \
data/kinetics400/kinetics_train_list_xlw \
data/kinetics400/kinetics_val_list_xlw \
--arch resnet50_3d_slowonly \
--dro 0.5 \
--mode 3D \
--t_length 8 \
--t_stride 8 \
--pretrained \
--epochs 110 \
--batch-size 96 \
--lr 0.02 \
--wd 0.0001 \
--lr_steps 50 80 100 \
--workers 16 \
```

## Testing
```Shell
python ./test_kaiming.py \
kinetics400 \
data/kinetics400/kinetics_val_list_xlw \
output/kinetics400_resnet50_3d_slowonly_3D_length8_stride8_dropout0.5/model_best.pth \
--arch resnet50_3d_slowonly \
--mode TSN+3D \
--batch_size 1 \
--num_segments 10 \
--input_size 256 \
--t_length 8 \
--t_stride 8 \
--dropout 0.5 \
--workers 12 \
--image_tmpl image_{:06d}.jpg \

```