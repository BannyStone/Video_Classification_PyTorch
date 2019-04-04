# CUDA_LAUNCH_BLOCKING=1 \
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# python main.py \
# kinetics400 \
# data/kinetics400/kinetics_train_list.txt \
# data/kinetics400/kinetics_val_list.txt \
# --arch gsv_resnet50_3d_v3 \
# --dro 0.2 \
# --mode 3D \
# --t_length 16 \
# --t_stride 4 \
# --epochs 112 \
# --batch-size 64 \
# --lr 0.001 \
# --lr_steps 60 90 106 \
# --workers 32 \
# --eval-freq 2 \
# --pretrained \
# --resume output/kinetics400_gsv_resnet50_3d_v3_3D_length16_stride4_dropout0.2/checkpoint_100epoch.pth \
# checkpoint_106epoch
python ./test.py \
kinetics400 \
data/kinetics400/kinetics_val_list.txt \
./output/kinetics400_gsv_resnet50_3d_v3_3D_length16_stride4_dropout0.2/model_best.pth \
--arch gsv_resnet50_3d_v3 \
--mode TSN+3D \
--batch_size 2 \
--num_segments 10 \
--t_length 16 \
--t_stride 4 \
--crop_fusion_type max \
--dropout 0.2 \
--workers 16 \
--image_tmpl image_{:06d}.jpg \
--save_scores ./output/kinetics400_gsv_resnet50_3d_v3_3D_length16_stride4_dropout0.2 \

# python ./test_kaiming.py \
# kinetics400 \
# data/kinetics400/kinetics_val_list.txt \
# ./output/kinetics400_gsv_resnet50_3d_v3_3D_length16_stride4_dropout0.2/model_best.pth \
# --arch gsv_resnet50_3d_v3 \
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
# --save_scores ./output/kinetics400_gsv_resnet50_3d_v3_3D_length16_stride4_dropout0.2 \
