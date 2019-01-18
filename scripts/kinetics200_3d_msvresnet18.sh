# CUDA_LAUNCH_BLOCKING=1 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python main.py \
kinetics200 \
data/kinetics200/kinetics200_train_list.txt \
data/kinetics200/kinetics200_val_list.txt \
--arch msv_resnet18 \
--dro 0.2 \
--mode 3D \
--new_size 128 \
--crop_size 112 \
--t_length 16 \
--t_stride 4 \
--epochs 130 \
--batch-size 128 \
--lr 0.01 \
--lr_steps 80 100 120 \
--workers 32 \
