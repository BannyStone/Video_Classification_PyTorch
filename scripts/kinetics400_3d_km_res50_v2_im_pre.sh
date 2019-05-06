# CUDA_LAUNCH_BLOCKING=1 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python main.py \
kinetics400 \
data/kinetics400/kinetics_train_list.txt \
data/kinetics400/kinetics_val_list.txt \
--arch km_resnet50_3d_v2_0init_tem_reciprocal16 \
--dro 0.2 \
--mode 3D \
--t_length 8 \
--t_stride 8 \
--pretrained \
--epochs 110 \
--batch-size 96 \
--lr 0.001 \
--lr_steps 60 90 100 \
--workers 16 \

# python ./test_kaiming.py \
# kinetics400 \
# data/kinetics400/kinetics_val_list.txt \
# output/kinetics400_resnet50_3d_v3_3D_length16_stride4_dropout0.2/model_best.pth \
# --arch resnet50_3d_v3 \
# --mode TSN+3D \
# --batch_size 1 \
# --num_segments 10 \
# --input_size 256 \
# --t_length 16 \
# --t_stride 4 \
# --crop_fusion_type avg \
# --dropout 0.2 \
# --workers 12 \
# --image_tmpl image_{:06d}.jpg \
# --save_scores output/kinetics400_resnet50_3d_v3_3D_length16_stride4_dropout0.2 \