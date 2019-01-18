# CUDA_LAUNCH_BLOCKING=1 \
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# python main.py \
# kinetics200 \
# data/kinetics200/kinetics200_train_list.txt \
# data/kinetics200/kinetics200_val_list.txt \
# --arch fst_resnet18_sd2_st1_x4 \
# --dro 0.2 \
# --mode 3D \
# --new_size 128 \
# --crop_size 112 \
# --t_length 16 \
# --t_stride 4 \
# --epochs 120 \
# --batch-size 224 \
# --lr 0.01 \
# --lr_steps 60 90 110 \
# --workers 20 \
# --resume output/kinetics200_fst_resnet18_sd2_st1_x4_3D_length16_stride4_dropout0.2/checkpoint_76epoch.pth \

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# python main.py \
# kinetics200 \
# data/kinetics200/kinetics200_train_list.txt \
# data/kinetics200/kinetics200_val_list.txt \
# --arch fst_resnet18_sd2_st4_x4 \
# --dro 0.2 \
# --mode 3D \
# --new_size 128 \
# --crop_size 112 \
# --t_length 16 \
# --t_stride 4 \
# --epochs 120 \
# --batch-size 224 \
# --lr 0.01 \
# --lr_steps 60 90 110 \
# --workers 20 \

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# python main.py \
# kinetics200 \
# data/kinetics200/kinetics200_train_list.txt \
# data/kinetics200/kinetics200_val_list.txt \
# --arch fst_resnet18_sd4_st1_x4 \
# --dro 0.2 \
# --mode 3D \
# --new_size 128 \
# --crop_size 112 \
# --t_length 16 \
# --t_stride 4 \
# --epochs 120 \
# --batch-size 224 \
# --lr 0.01 \
# --lr_steps 60 90 110 \
# --workers 20 \

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# python main.py \
# kinetics200 \
# data/kinetics200/kinetics200_train_list.txt \
# data/kinetics200/kinetics200_val_list.txt \
# --arch fst_resnet18_sd4_st4_x4 \
# --dro 0.2 \
# --mode 3D \
# --new_size 128 \
# --crop_size 112 \
# --t_length 16 \
# --t_stride 4 \
# --epochs 120 \
# --batch-size 224 \
# --lr 0.01 \
# --lr_steps 60 90 110 \
# --workers 20 \

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python main.py \
kinetics200 \
data/kinetics200/kinetics200_train_list.txt \
data/kinetics200/kinetics200_val_list.txt \
--arch fst_resnet18_sf5_st1_x4 \
--dro 0.2 \
--mode 3D \
--new_size 128 \
--crop_size 112 \
--t_length 16 \
--t_stride 4 \
--epochs 120 \
--batch-size 224 \
--lr 0.01 \
--lr_steps 60 90 110 \
--workers 20 \