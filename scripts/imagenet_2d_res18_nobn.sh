# CUDA_LAUNCH_BLOCKING=1 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python main_imagenet.py \
imagenet \
placeholder \
placeholder \
--arch resnet18_nobn \
--epochs 100 \
--batch-size 1024 \
--lr 0.2 \
--lr_steps 30 60 90 \
--workers 16 \
--weight-decay 0.0001 \
--eval-freq 1 \
