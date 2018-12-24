# Video-Classification-Pytorch

*Still in development*.

This is a repository containing 3D models and 2D models for video classification. The code is based on PyTorch.
Util now, it supports the following datasets:
Kinetics-400, MiniKinetics, UCF101

## Datasets
*Note: our Kinetics-400 has 240439 training videos and 19796 validation videos*.

## Results

We report the baselines with ResNet-50 backbone on Kinetics-400 validation set as below (training data is Kinetics-400 training set).
All the models are trained in one single server with 8 GTX 1080 Ti GPUs.

| <sub>network</sub> | <sub>pretrain data</sub> | <sub>input frames</sub> | <sub>sampling stride</sub> | <sub>backbone</sub> | <sub>top1</sub> | <sub>top5</sub> |
| ------------------ | ------------------ | ------------------ | ------------------ | ------------------ | ------------------ | ------------------ |
| ResNet50-I3D | ImageNet-1K | 16 | 4 | ResNet50 | 73.45 | 91.11 |


## Training
All the training scripts with ResNet-50 backbone are here:
```Shell
cd scripts
```

We show one script here *kinetics400_3d_res101_v1_im_pre.sh*:
```Shell
python main.py kinetics400 data/kinetics400/kinetics_train_list.txt data/kinetics400/kinetics_val_list.txt --arch resnet50_3d_v1 --dro 0.2 --mode 3D --t_length 16 --t_stride 4 --pretrained --epochs 110 --batch-size 64 --lr 0.001 --lr_steps 60 90 100 --workers 32
```

## Testing
```Shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python ./test.py \
kinetics400 \
data/kinetics400/kinetics_val_list.txt \
./output/kinetics400_resnet50_3d_3D_length16_stride4_dropout0.2/model_best.pth \
--arch resnet50_3d \
--mode TSN+3D \
--batch_size 2 \
--num_segments 15 \
--test_crops 10 \
--t_length 16 \
--t_stride 4 \
--crop_fusion_type max \
--dropout 0.2 \
--workers 12 \
--save_scores ./output/kinetics400_resnet50_3d_3D_length16_stride4_dropout0.2
```
