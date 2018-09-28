CUDA_VISIBLE_DEVICES=0,1,2,3 \
python main.py \
kinetics400 \
data/kinetics400/kinetics_train_list.txt \
data/kinetics400/kinetics_val_list.txt \
--arch mnet2 \
--dro 0.2 \
--mode 2D \
--t_length 32 \
--t_stride 2 \
--pretrained \
--epochs 75 \
--batch-size 256 \
--lr 0.01 \
--lr_steps 24 58 70 \
--workers 16 \
