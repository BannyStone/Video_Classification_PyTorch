CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python main.py \
kinetics400 \
data/kinetics400/kinetics_train_list_xlw \
data/kinetics400/kinetics_val_list_xlw \
--arch resnet50_3d_p2stream \
--dro 0.5 \
--mode 3D \
--t_length 8 \
--t_stride 8 \
--pretrained \
--epochs 110 \
--batch-size 96 \
--lr 0.02 \
--wd 0.0001 \
--lr_steps 50 80 100 \
--workers 16 \