CUDA_VISIBLE_DEVICES=4,5,6,7 \
python main.py \
kinetics400 \
data/kinetics400/kinetics_train_list.txt \
data/kinetics400/kinetics_val_list.txt \
/home/leizhou/CVPR2019/vid_cls/data/kinetics400 \
--arch resnet50 \
--dro 0.2 \
--mode 2D \
--t_length 32 \
--t_stride 2 \
--pretrained \
--epochs 95 \
--batch-size 256 \
--lr 0.01 \
--lr_steps 40 80 90 \
--workers 16
