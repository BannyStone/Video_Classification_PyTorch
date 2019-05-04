CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python main.py \
kinetics200 \
data/kinetics200/kinetics200_train_list.txt \
data/kinetics200/kinetics200_val_list.txt \
--arch km_resnet26_3d_v2_1 \
--dro 0.4 \
--mode 3D \
--t_length 8 \
--t_stride 8 \
--epochs 80 \
--batch-size 128 \
--lr 0.01 \
--lr_steps 30 60 \
--workers 32 \
--eval-freq 2 \
--pretrained \
--pretrained_model models/resnet26.pth \
