# CUDA_LAUNCH_BLOCKING=1 \
CUDA_VISIBLE_DEVICES=4,5,6,7 \
python main_20bn.py \
sthsth_v1 \
data/sthsth_v1/sthv1_train_list.txt \
data/sthsth_v1/sthv1_val_list.txt \
--arch km_resnet50_3d_v2_0init_tem_reciprocal16 \
--dro 0.4 \
--mode 3D \
--t_length 8 \
--t_stride 8 \
--epochs 60 \
--batch-size 48 \
--lr 0.01 \
--wd 0.0001 \
--lr_steps 40 50 \
--workers 16 \
--image_tmpl "{:05d}.jpg" \
--pretrained \
--resume output/sthsth_v1_km_resnet50_3d_v2_0init_tem_reciprocal16_3D_length8_stride8_dropout0.4/checkpoint_40epoch.pth

# CUDA_VISIBLE_DEVICES=4,5,6,7 \
# python finetune_new_20bn.py \
# sthsth_v1 \
# data/sthsth_v1/sthv1_train_list.txt \
# data/sthsth_v1/sthv1_val_list.txt \
# --arch km_resnet50_3d_v2_0init_tem_reciprocal16 \
# --dro 0.6 \
# --mode 3D \
# --t_length 8 \
# --t_stride 4 \
# --epochs 80 \
# --batch-size 48 \
# --lr 0.01 \
# --wd 0.0001 \
# --lr_steps 40 60 70 \
# --workers 16 \
# --image_tmpl "{:05d}.jpg" \
# --pretrained \
# --resume output/sthsth_v1_km_resnet50_3d_v2_0init_tem_reciprocal16_3D_length8_stride8_dropout0.4/checkpoint_20epoch.pth

# CUDA_VISIBLE_DEVICES=4,5,6,7 \
# python main_20bn.py \
# sthsth_v1 \
# data/sthsth_v1/sthv1_train_list.txt \
# data/sthsth_v1/sthv1_val_list.txt \
# --arch km_resnet26_3d_v2_0init_tem_reciprocal16 \
# --dro 0.4 \
# --mode 3D \
# --t_length 8 \
# --t_stride 8 \
# --epochs 50 \
# --batch-size 80 \
# --lr 0.01 \
# --wd 0.0001 \
# --lr_steps 30 40 \
# --workers 16 \
# --image_tmpl "{:05d}.jpg" \
# --pretrained \
# --pretrained_model models/resnet26.pth \
# --resume output/sthsth_v1_km_resnet26_3d_v2_0init_tem_reciprocal16_3D_length8_stride8_dropout0.4/checkpoint_20epoch.pth

# CUDA_VISIBLE_DEVICES=4,5,6,7 \
# python main_20bn.py \
# sthsth_v1 \
# data/sthsth_v1/sthv1_train_list.txt \
# data/sthsth_v1/sthv1_val_list.txt \
# --arch km_resnet50_3d_v2_0init_tem_reciprocal4 \
# --dro 0.4 \
# --mode 3D \
# --t_length 8 \
# --t_stride 8 \
# --epochs 60 \
# --batch-size 48 \
# --lr 0.01 \
# --wd 0.0001 \
# --lr_steps 20 40 50 \
# --workers 16 \
# --image_tmpl "{:05d}.jpg" \
# --pretrained \

# python ./test_kaiming.py \
# sthsth_v1 \
# data/sthsth_v1/sthv1_val_list.txt \
# ./output/sthsth_v1_pib_resnet50_3d_slow_3D_length8_stride8_dropout0.4/model_best.pth \
# --arch pib_resnet50_3d_slow \
# --mode TSN+3D \
# --batch_size 1 \
# --num_segments 4 \
# --input_size 256 \
# --t_length 8 \
# --t_stride 8 \
# --crop_fusion_type avg \
# --dropout 0.4 \
# --workers 16 \
# --image_tmpl {:05d}.jpg \
# --save_scores ./output/sthsth_v1_pib_resnet50_3d_slow_3D_length8_stride8_dropout0.4 \
