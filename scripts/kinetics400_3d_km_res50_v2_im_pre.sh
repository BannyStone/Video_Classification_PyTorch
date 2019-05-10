# CUDA_LAUNCH_BLOCKING=1 \
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# python main.py \
# kinetics400 \
# data/kinetics400/kinetics_train_list.txt \
# data/kinetics400/kinetics_val_list.txt \
# --arch km_resnet50_3d_v2_0init_tem_reciprocal16 \
# --dro 0.2 \
# --mode 3D \
# --t_length 8 \
# --t_stride 8 \
# --pretrained \
# --epochs 110 \
# --batch-size 96 \
# --lr 0.001 \
# --lr_steps 60 90 \
# --workers 16 \
# --resume output/kinetics400_km_resnet50_3d_v2_0init_tem_reciprocal16_3D_length8_stride8_dropout0.2/checkpoint_76epoch.pth

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# python main.py \
# kinetics400 \
# data/kinetics400/kinetics_train_list.txt \
# data/kinetics400/kinetics_val_list.txt \
# --arch km_resnet50_3d_v2_0init_tem_reciprocal16 \
# --dro 0.2 \
# --mode 3D \
# --t_length 8 \
# --t_stride 8 \
# --pretrained \
# --epochs 100 \
# --batch-size 96 \
# --lr 0.001 \
# --lr_steps 60 80 \
# --workers 16 \
# --resume output/kinetics400_km_resnet50_3d_v2_0init_tem_reciprocal16_3D_length8_stride8_dropout0.2/checkpoint_80epoch.pth

python ./test_kaiming.py \
kinetics400 \
data/kinetics400/kinetics_val_list.txt \
output/kinetics400_km_resnet50_3d_v2_0init_tem_reciprocal16_3D_length8_stride8_dropout0.2/past_best/model_best.pth \
--arch km_resnet50_3d_v2_0init_tem_reciprocal16 \
--mode TSN+3D \
--batch_size 1 \
--num_segments 10 \
--input_size 256 \
--t_length 8 \
--t_stride 8 \
--crop_fusion_type max \
--dropout 0.2 \
--workers 12 \
--image_tmpl image_{:06d}.jpg \
--save_scores output/kinetics400_km_resnet50_3d_v2_0init_tem_reciprocal16_3D_length8_stride8_dropout0.2 \