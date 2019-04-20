# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# python main.py \
# kinetics200 \
# data/kinetics200/kinetics200_train_list.txt \
# data/kinetics200/kinetics200_val_list.txt \
# --arch gps_base_resnet26_3d_v1 \
# --dro 0.2 \
# --mode 3D \
# --t_length 16 \
# --t_stride 4 \
# --epochs 70 \
# --batch-size 64 \
# --lr 0.001 \
# --lr_steps 40 60 \
# --workers 32 \
# --eval-freq 2 \
# --pretrained \
# --pretrained_model models/resnet26.pth \

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python main.py \
kinetics200 \
data/kinetics200/kinetics200_train_list.txt \
data/kinetics200/kinetics200_val_list.txt \
--arch gps_resnet26_3d_v1 \
--dro 0.2 \
--mode 3D \
--t_length 16 \
--t_stride 4 \
--epochs 70 \
--batch-size 64 \
--lr 0.001 \
--lr_steps 40 60 \
--workers 32 \
--eval-freq 2 \
--pretrained \
--pretrained_model models/resnet26.pth \

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# python main.py \
# kinetics200 \
# data/kinetics200/kinetics200_train_list.txt \
# data/kinetics200/kinetics200_val_list.txt \
# --arch gps_base_resnet26_3d_v2 \
# --dro 0.2 \
# --mode 3D \
# --t_length 16 \
# --t_stride 4 \
# --epochs 70 \
# --batch-size 64 \
# --lr 0.001 \
# --lr_steps 40 60 \
# --workers 32 \
# --eval-freq 2 \
# --pretrained \
# --pretrained_model models/resnet26.pth \

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# python main.py \
# kinetics200 \
# data/kinetics200/kinetics200_train_list.txt \
# data/kinetics200/kinetics200_val_list.txt \
# --arch gps_resnet26_3d_v2 \
# --dro 0.2 \
# --mode 3D \
# --t_length 16 \
# --t_stride 4 \
# --epochs 70 \
# --batch-size 64 \
# --lr 0.001 \
# --lr_steps 40 60 \
# --workers 32 \
# --eval-freq 2 \
# --pretrained \
# --pretrained_model models/resnet26.pth \