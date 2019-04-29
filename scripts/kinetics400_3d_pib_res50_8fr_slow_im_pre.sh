# CUDA_LAUNCH_BLOCKING=1 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python main.py \
kinetics400 \
data/kinetics400/kinetics_train_list.txt \
data/kinetics400/kinetics_val_list.txt \
--arch pib_resnet50_3d_slow \
--dro 0.4 \
--mode 3D \
--t_length 8 \
--t_stride 8 \
--epochs 110 \
--batch-size 96 \
--lr 0.01 \
--wd 0.0001 \
--lr_steps 40 80 100 \
--workers 32 \
--pretrained \