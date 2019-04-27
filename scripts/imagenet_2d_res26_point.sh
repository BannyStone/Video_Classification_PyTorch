# CUDA_LAUNCH_BLOCKING=1 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python main_imagenet.py \
imagenet \
placeholder \
placeholder \
--arch resnet26_point \
--epochs 100 \
--batch-size 512 \
--lr 0.1 \
--lr_steps 30 50 70 90 \
--workers 20 \
--weight-decay 0.0001 \
--eval-freq 1 \