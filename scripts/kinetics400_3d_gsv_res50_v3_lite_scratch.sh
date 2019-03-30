CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python main.py \
kinetics400 \
data/kinetics400/kinetics_train_list.txt \
data/kinetics400/kinetics_val_list.txt \
--arch gsv_resnet50_3d_v3_lite \
--dro 0.2 \
--mode 3D \
--t_length 8 \
--t_stride 8 \
--epochs 100 \
--batch-size 112 \
--lr 0.01 \
--lr_steps 40 60 90 \
--workers 32 \
