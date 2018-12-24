# CUDA_LAUNCH_BLOCKING=1 \
CUDA_VISIBLE_DEVICES=0,1 \
python main.py \
ucf101 \
data/ucf101/ucf101_train_split1_list.txt \
data/ucf101/ucf101_val_split1_list.txt \
--arch resnet50_3d \
--dro 0.4 \
--mode 3D \
--t_length 16 \
--t_stride 4 \
--epochs 90 \
--batch-size 8 \
--lr 0.01 \
--lr_steps 40 80 \
--workers 16 \
--image_tmpl img_{:05d}.jpg \
-ef 5 \
# --resume output/ucf101_resnet50_3d_3D_length16_stride4_dropout0.4/checkpoint_10epoch.pth \
#--pretrained \
#--pretrained_model models/kinetics400_pre_3d_pre_2d/kinetics400_pre_3d_pre_2d.pth \

