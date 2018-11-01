# CUDA_LAUNCH_BLOCKING=1 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python main_shadow.py \
kinetics200 \
data/kinetics200/kinetics200_train_list.txt \
data/kinetics200/kinetics200_val_list.txt \
--arch resnet50_3d \
--shadow \
--dro 0.2 \
--mode 3D \
--t_length 16 \
--t_stride 4 \
--num_segments 3 \
--pretrained \
--epochs 95 \
--batch-size 56 \
--lr 0.001 \
--lr_steps 40 80 90 \
--workers 16 \
--image_tmpl image_{:06d}.jpg \
-ef 1 \