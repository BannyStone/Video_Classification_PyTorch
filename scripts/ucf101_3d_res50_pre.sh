# CUDA_LAUNCH_BLOCKING=1 \
CUDA_VISIBLE_DEVICES=3,5 \
python main.py \
ucf101 \
data/ucf101/ucf101_train_split1_list.txt \
data/ucf101/ucf101_val_split1_list.txt \
--arch resnet50_3d \
--dro 0.2 \
--mode 3D \
--t_length 16 \
--t_stride 4 \
--pretrained \
--epochs 95 \
--batch-size 8 \
--lr 0.001 \
--lr_steps 40 80 90 \
--workers 16 \
--image_tmpl img_{:05d}.jpg