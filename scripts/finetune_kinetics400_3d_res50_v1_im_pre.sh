# # CUDA_LAUNCH_BLOCKING=1 \
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# python finetune.py \
# kinetics400 \
# data/kinetics400/kinetics_train_list.txt \
# data/kinetics400/kinetics_val_list.txt \
# --arch resnet50_3d_v1 \
# --dro 0.2 \
# --mode 3D \
# --t_length 61 \
# --t_stride 4 \
# --pretrained \
# --pretrained_model models/resnet50_3d_v1.pth \
# --epochs 20 \
# --batch-size 32 \
# --lr 0.0025 \
# --lr_steps 10 16 \
# --workers 28 \
# --resume output/kinetics400_resnet50_3d_v1_3D_length61_stride4_dropout0.2/checkpoint_16epoch.pth \

# CUDA_LAUNCH_BLOCKING=1 \
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# python finetune.py \
# kinetics400 \
# data/kinetics400/kinetics_train_list.txt \
# data/kinetics400/kinetics_val_list.txt \
# --arch resnet50_3d_v1 \
# --dro 0.2 \
# --mode 3D \
# --t_length 16 \
# --t_stride 16 \
# --pretrained \
# --pretrained_model models/resnet50_3d_v1.pth \
# --epochs 20 \
# --batch-size 64 \
# --lr 0.002 \
# --lr_steps 10 16 \
# --workers 32 \

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# python finetune.py \
# kinetics400 \
# data/kinetics400/kinetics_train_list.txt \
# data/kinetics400/kinetics_val_list.txt \
# --arch resnet50_3d_v1 \
# --dro 0.2 \
# --mode 3D \
# --t_length 8 \
# --t_stride 4 \
# --pretrained \
# --pretrained_model models/resnet50_3d_v1.pth \
# --epochs 10 \
# --batch-size 64 \
# --lr 0.0025 \
# --lr_steps 6 \
# --workers 20 \

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# python finetune.py \
# kinetics400 \
# data/kinetics400/kinetics_train_list.txt \
# data/kinetics400/kinetics_val_list.txt \
# --arch resnet50_3d_v1 \
# --dro 0.2 \
# --mode 3D \
# --t_length 31 \
# --t_stride 2 \
# --pretrained \
# --pretrained_model models/resnet50_3d_v1.pth \
# --epochs 10 \
# --batch-size 64 \
# --lr 0.0025 \
# --lr_steps 6 \
# --workers 20 \

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# python finetune.py \
# kinetics400 \
# data/kinetics400/kinetics_train_list.txt \
# data/kinetics400/kinetics_val_list.txt \
# --arch resnet50_3d_v1 \
# --dro 0.2 \
# --mode 3D \
# --t_length 9 \
# --t_stride 8 \
# --pretrained \
# --pretrained_model models/resnet50_3d_v1.pth \
# --epochs 10 \
# --batch-size 64 \
# --lr 0.0025 \
# --lr_steps 6 \
# --workers 20 \

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# python finetune.py \
# kinetics400 \
# data/kinetics400/kinetics_train_list.txt \
# data/kinetics400/kinetics_val_list.txt \
# --arch resnet50_3d_v1 \
# --dro 0.2 \
# --mode 3D \
# --t_length 5 \
# --t_stride 16 \
# --pretrained \
# --pretrained_model models/resnet50_3d_v1.pth \
# --epochs 10 \
# --batch-size 64 \
# --lr 0.0025 \
# --lr_steps 6 \
# --workers 20 \

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python finetune.py \
kinetics400 \
data/kinetics400/kinetics_train_list.txt \
data/kinetics400/kinetics_val_list.txt \
--arch resnet50_3d_v1 \
--dro 0.2 \
--mode 3D \
--t_length 16 \
--t_stride 2 \
--pretrained \
--pretrained_model models/resnet50_3d_v1.pth \
--epochs 10 \
--batch-size 64 \
--lr 0.0025 \
--lr_steps 6 \
--workers 20 \