CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python main.py \
kinetics400 \
data/kinetics400/kinetics_train_list.txt \
data/kinetics400/kinetics_val_list.txt \
--arch ada_resnet50_3d_v1_1_1 \
--dro 0.2 \
--mode 3D \
--t_length 16 \
--t_stride 4 \
--epochs 110 \
--batch-size 64 \
--lr 0.001 \
--lr_steps 60 90 \
--workers 32 \
--eval-freq 2 \
--pretrained \
