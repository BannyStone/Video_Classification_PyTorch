# CUDA_LAUNCH_BLOCKING=1 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python main.py \
kinetics400 \
data/kinetics400/kinetics_train_list.txt \
data/kinetics400/kinetics_val_list.txt \
--arch resnet101_3d_v1 \
--dro 0.2 \
--mode 3D \
--t_length 16 \
--t_stride 4 \
--pretrained \
--epochs 110 \
--batch-size 56 \
--lr 0.001 \
--lr_steps 60 90 100 \
--workers 16 \
--resume output/kinetics400_resnet101_3d_v1_3D_length16_stride4_dropout0.2/checkpoint_28epoch.pth \
