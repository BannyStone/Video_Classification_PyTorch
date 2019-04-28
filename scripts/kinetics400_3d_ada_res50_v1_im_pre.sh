# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# python main.py \
# kinetics400 \
# data/kinetics400/kinetics_train_list.txt \
# data/kinetics400/kinetics_val_list.txt \
# --arch ada_resnet50_3d_v1_1_1 \
# --dro 0.2 \
# --mode 3D \
# --t_length 16 \
# --t_stride 4 \
# --epochs 110 \
# --batch-size 64 \
# --lr 0.001 \
# --lr_steps 60 90 \
# --workers 32 \
# --eval-freq 2 \
# --pretrained \
# --resume output/kinetics400_ada_resnet50_3d_v1_1_1_3D_length16_stride4_dropout0.2/checkpoint_56epoch.pth

python ./test.py \
kinetics400 \
data/kinetics400/kinetics_val_list.txt \
./output/kinetics400_ada_resnet50_3d_v1_1_1_3D_length16_stride4_dropout0.2/model_best.pth \
--arch ada_resnet50_3d_v1_1_1 \
--mode TSN+3D \
--batch_size 2 \
--num_segments 10 \
--t_length 16 \
--t_stride 4 \
--crop_fusion_type avg \
--dropout 0.2 \
--workers 16 \
--image_tmpl image_{:06d}.jpg \
--save_scores ./output/kinetics400_ada_resnet50_3d_v1_1_1_3D_length16_stride4_dropout0.2
