# CUDA_LAUNCH_BLOCKING=1 \
CUDA_VISIBLE_DEVICES=4,5,6,7 \
python main_imagenet.py \
imagenet \
placeholder \
placeholder \
--arch resnet26 \
--epochs 100 \
--batch-size 512 \
--lr 0.1 \
--lr_steps 30 50 70 90 \
--workers 20 \
--weight-decay 0.0001 \
--eval-freq 1 \
