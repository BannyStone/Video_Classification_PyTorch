# CUDA_LAUNCH_BLOCKING=1 \
CUDA_VISIBLE_DEVICES=6,7 \
python main_shadow_test.py \
ucf101 \
data/ucf101/ucf101_train_split1_list.txt \
data/ucf101/ucf101_val_split1_list.txt \
--arch resnet50_3d \
--shadow \
--dro 0 \
--mode 3D \
--t_length 16 \
--t_stride 4 \
--num_segments 3 \
--pretrained \
--pretrained_model models/kinetics400_pre_3d_pre_2d/kinetics400_pre_3d_pre_2d.pth \
--epochs 40 \
--batch-size 8 \
--lr 0.001 \
--lr_steps 15 25 35 \
--workers 16 \
--image_tmpl img_{:05d}.jpg \
-ef 1 \
