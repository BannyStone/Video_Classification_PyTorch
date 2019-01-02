# Video-Classification-Pytorch

***Still in development***.

This is a repository containing 3D models and 2D models for video classification. The code is based on PyTorch.
Until now, it supports the following datasets:
Kinetics-400, Mini-Kinetics-200, UCF101

## Results

### Kinetics-400

We report the baselines with ResNet-50 backbone on Kinetics-400 validation set as below (all models are trained on training set).
All the models are trained in one single server with 8 GTX 1080 Ti GPUs.

| <sub>network</sub> | <sub>pretrain data</sub> | <sub>spatial resolution</sub> | <sub>input frames</sub> | <sub>sampling stride</sub> | <sub>backbone</sub> | <sub>top1</sub> | <sub>top5</sub> |
| ------------------ | ------------------ | ------------------ | ------------------ | ------------------ | ------------------ | ------------------ | ------------------ |
| <sub>ResNet50-I3D</sub> | <sub>ImageNet-1K</sub> | <sub>224x224</sub> | <sub>16</sub> | <sub>4</sub> | <sub>ResNet50</sub> | <sub>73.45</sub> | <sub>91.11</sub> |
| <sub>ResNet101-I3D</sub> | <sub>ImageNet-1k</sub> | <sub>224x224</sub> | <sub>16</sub> | <sub>4</sub> | <sub>ResNet101</sub> | <sub>74.43</sub> | <sub>91.84</sub> |


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
- Our Kinetics-400 has 240439 training videos and 19796 validation videos.
- In train and validation lists for all datasets, each line represents one video where the first element is the video frame directory, the second element is the number of frames and the third element is the index of class. Please prepare your own list accordingly because different video parsing method may lead to different frame numbers.
- This code can read the image files in each video frame folder according to the image template argument *image_tmpl*, such as *image_{:06d}.jpg*.

## Training
Execute training script:
```Shell
./scripts/kinetics400_3d_res50_v1_im_pre.sh
```

We show script *kinetics400_3d_res50_v1_im_pre.sh* here:
```Shell
python main.py kinetics400 data/kinetics400/kinetics_train_list.txt data/kinetics400/kinetics_val_list.txt --arch resnet50_3d_v1 --dro 0.2 --mode 3D --t_length 16 --t_stride 4 --pretrained --epochs 110 --batch-size 64 --lr 0.001 --lr_steps 60 90 100 --workers 32
```

## Testing
```Shell
python ./test.py kinetics400 data/kinetics400/kinetics_val_list.txt ./output/kinetics400_resnet50_3d_v1_3D_length16_stride4_dropout0.2/model_best.pth --arch resnet50_3d_v1 --mode TSN+3D --batch_size 2 --num_segments 10 --test_crops 10 --t_length 16 --t_stride 4 --crop_fusion_type avg --dropout 0.2 --workers 16 --image_tmpl image_{:06d}.jpg --save_scores ./output/kinetics400_resnet50_3d_v1_3D_length16_stride4_dropout0.2

```