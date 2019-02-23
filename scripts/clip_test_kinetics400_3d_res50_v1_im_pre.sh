CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python clip_test.py \
kinetics400 \
data/kinetics400/kinetics_train_list.txt \
data/kinetics400/kinetics_val_list.txt \
--arch resnet50_3d_v1 \
--dro 0.2 \
--mode 3D \
--t_length 61 \
--t_stride 1 \
--pretrained \
--batch-size 64 \
--workers 10 \
--resume /home/leizhou/Research/vid_cls/output/kinetics400_resnet50_3d_v1_3D_length16_stride4_dropout0.2/model_best.pth
