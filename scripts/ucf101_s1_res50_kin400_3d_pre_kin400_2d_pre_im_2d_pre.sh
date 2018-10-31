# CUDA_LAUNCH_BLOCKING=1 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python main.py \
ucf101 \
data/ucf101/ucf101_train_split1_list.txt \
data/ucf101/ucf101_val_split1_list.txt \
--arch resnet50_3d \
--dro 0.4 \
--mode 3D \
--t_length 16 \
--t_stride 4 \
--pretrained \
--pretrained_model models/kinetics400_pre_3d_pre_2d/kinetics400_pre_3d_pre_2d.pth \
--epochs 25 \
--batch-size 56 \
--lr 0.001 \
--lr_steps 10 20 \
--workers 16 \
--image_tmpl img_{:05d}.jpg \
-ef 1 \
