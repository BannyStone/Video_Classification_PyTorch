CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python main.py \
kinetics400 \
data/kinetics400/kinetics_train_list.txt \
data/kinetics400/kinetics_val_list.txt \
--arch resnet50 \
--dro 0.2 \
--mode 2D \
--t_length 32 \
--t_stride 2 \
--pretrained \
--epochs 120 \
--batch-size 512 \
--lr 0.01 \
--lr_steps 40 80 100 \
--workers 32 \
--resume /home/leizhou/CVPR2019/vid_cls/output/kinetics400_resnet50_2D_length1_stride2_dropout0.2/checkpoint_90epoch.pth