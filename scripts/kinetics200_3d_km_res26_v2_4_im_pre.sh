# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# python main.py \
# kinetics200 \
# data/kinetics200/kinetics200_train_list.txt \
# data/kinetics200/kinetics200_val_list.txt \
# --arch km_resnet26_3d_v2_4 \
# --dro 0.4 \
# --mode 3D \
# --t_length 8 \
# --t_stride 8 \
# --epochs 80 \
# --batch-size 128 \
# --lr 0.01 \
# --lr_steps 28 60 \
# --workers 16 \
# --eval-freq 2 \
# --pretrained \
# --pretrained_model models/resnet26.pth \
# --resume output/kinetics200_km_resnet26_3d_v2_4_3D_length8_stride8_dropout0.4/checkpoint_40epoch.pth

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# python main.py \
# kinetics200 \
# data/kinetics200/kinetics200_train_list.txt \
# data/kinetics200/kinetics200_val_list.txt \
# --arch km_resnet26_3d_v2_3_0init_tem4 \
# --dro 0.4 \
# --mode 3D \
# --t_length 8 \
# --t_stride 8 \
# --epochs 80 \
# --batch-size 128 \
# --lr 0.01 \
# --lr_steps 30 60 \
# --workers 16 \
# --eval-freq 2 \
# --pretrained \
# --pretrained_model models/resnet26.pth \

# python ./test_kaiming.py \
# kinetics200 \
# data/kinetics200/kinetics200_val_list.txt \
# ./output/kinetics200_km_resnet26_3d_v2_3_3D_length8_stride8_dropout0.4/model_best.pth \
# --arch km_resnet26_3d_v2_3 \
# --mode TSN+3D \
# --batch_size 1 \
# --num_segments 10 \
# --input_size 256 \
# --t_length 8 \
# --t_stride 8 \
# --crop_fusion_type avg \
# --dropout 0.4 \
# --workers 16 \
# --image_tmpl image_{:06d}.jpg \
# --save_scores ./output/kinetics200_km_resnet26_3d_v2_3_3D_length8_stride8_dropout0.4 \

# python ./test_kaiming.py \
# kinetics200 \
# data/kinetics200/kinetics200_val_list.txt \
# ./output/kinetics200_km_resnet26_3d_v2_4_3D_length8_stride8_dropout0.4/model_best.pth \
# --arch km_resnet26_3d_v2_4 \
# --mode TSN+3D \
# --batch_size 1 \
# --num_segments 10 \
# --input_size 256 \
# --t_length 8 \
# --t_stride 8 \
# --crop_fusion_type avg \
# --dropout 0.4 \
# --workers 16 \
# --image_tmpl image_{:06d}.jpg \
# --save_scores ./output/kinetics200_km_resnet26_3d_v2_4_3D_length8_stride8_dropout0.4 \

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# python main.py \
# kinetics200 \
# data/kinetics200/kinetics200_train_list.txt \
# data/kinetics200/kinetics200_val_list.txt \
# --arch pib_resnet26_3d_v1_1 \
# --dro 0.4 \
# --mode 3D \
# --t_length 8 \
# --t_stride 8 \
# --epochs 80 \
# --batch-size 128 \
# --lr 0.01 \
# --lr_steps 30 60 \
# --workers 32 \
# --eval-freq 2 \
# --pretrained \
# --pretrained_model models/resnet26.pth \
# --resume output/kinetics200_pib_resnet26_3d_v1_1_3D_length8_stride8_dropout0.4/checkpoint_72epoch.pth

# python ./test_kaiming.py \
# kinetics200 \
# data/kinetics200/kinetics200_val_list.txt \
# ./output/kinetics200_pib_resnet26_3d_v1_1_3D_length8_stride8_dropout0.4/model_best.pth \
# --arch pib_resnet26_3d_v1_1 \
# --mode TSN+3D \
# --batch_size 1 \
# --num_segments 10 \
# --input_size 256 \
# --t_length 8 \
# --t_stride 8 \
# --crop_fusion_type avg \
# --dropout 0.4 \
# --workers 16 \
# --image_tmpl image_{:06d}.jpg \
# --save_scores ./output/kinetics200_pib_resnet26_3d_v1_1_3D_length8_stride8_dropout0.4 \

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# python main.py \
# kinetics200 \
# data/kinetics200/kinetics200_train_list.txt \
# data/kinetics200/kinetics200_val_list.txt \
# --arch km_resnet26_3d_v2_4_0init_tem4 \
# --dro 0.4 \
# --mode 3D \
# --t_length 8 \
# --t_stride 8 \
# --epochs 80 \
# --batch-size 128 \
# --lr 0.01 \
# --lr_steps 30 60 \
# --workers 16 \
# --eval-freq 2 \
# --pretrained \
# --pretrained_model models/resnet26.pth \
# --resume output/kinetics200_km_resnet26_3d_v2_4_0init_tem4_3D_length8_stride8_dropout0.4/checkpoint_8epoch.pth \

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python main.py \
kinetics200 \
data/kinetics200/kinetics200_train_list.txt \
data/kinetics200/kinetics200_val_list.txt \
--arch km_resnet26_3d_v2_4_random_init_tem4 \
--dro 0.4 \
--mode 3D \
--t_length 8 \
--t_stride 8 \
--epochs 80 \
--batch-size 128 \
--lr 0.01 \
--lr_steps 30 60 \
--workers 16 \
--eval-freq 2 \
--pretrained \
--pretrained_model models/resnet26.pth \