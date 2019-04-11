CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python main.py \
kinetics200 \
data/kinetics200/kinetics200_train_list.txt \
data/kinetics200/kinetics200_val_list.txt \
--arch gsv_resnet26_3d_v3 \
--dro 0.2 \
--mode 3D \
--t_length 16 \
--t_stride 4 \
--epochs 56 \
--batch-size 80 \
--lr 0.001 \
--lr_steps 30 50 \
--workers 32 \
--eval-freq 2 \
--pretrained \
--pretrained_model models/resnet26.pth
