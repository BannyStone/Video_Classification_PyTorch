# CUDA_LAUNCH_BLOCKING=1 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python main_imagenet.py \
imagenet \
placeholder \
placeholder \
--arch resnet26_sc \
--epochs 100 \
--batch-size 512 \
--lr 0.1 \
--lr_steps 30 50 70 90 \
--workers 20 \
--weight-decay 0.0001 \
--eval-freq 1 \
--resume output/imagenet_resnet26_sc_3D_length32_stride2_dropout0.2/checkpoint_30epoch.pth \