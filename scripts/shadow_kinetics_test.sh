# CUDA_LAUNCH_BLOCKING=1 \
CUDA_VISIBLE_DEVICES=0,1 \
python main_shadow.py \
kinetics400 \
data/kinetics400/kinetics_train_list.txt \
data/kinetics400/kinetics_val_list.txt \
--arch resnet50_3d \
--shadow \
--dro 0.2 \
--mode 3D \
--t_length 16 \
--t_stride 4 \
--num_segments 3 \
--pretrained \
--epochs 95 \
--batch-size 2 \
--lr 0.001 \
--lr_steps 40 80 90 \
--workers 8 \
--image_tmpl img_{:05d}.jpg \
