# CUDA_LAUNCH_BLOCKING=1 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python main_imagenet.py \
imagenet \
placeholder \
placeholder \
--arch gsv_resnet50_2d_v3 \
--epochs 120 \
--batch-size 512 \
--lr 0.05 \
--lr_steps 40 70 100 \
--workers 20 \
# --pretrained \
# --pretrained_model models/se_resnet50.pth
