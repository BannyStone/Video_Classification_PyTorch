# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# python main.py \
# kinetics200 \
# data/kinetics200/kinetics200_train_list.txt \
# data/kinetics200/kinetics200_val_list.txt \
# --arch km_resnet26_3d_v2_0init_tem_reciprocal16 \
# --dro 0.4 \
# --mode 3D \
# --t_length 8 \
# --t_stride 8 \
# --epochs 70 \
# --batch-size 128 \
# --lr 0.01 \
# --lr_steps 40 60 \
# --workers 16 \
# --eval-freq 2 \
# --pretrained \
# --pretrained_model models/resnet26.pth \

# python ./test_kaiming.py \
# kinetics200 \
# data/kinetics200/kinetics200_val_list.txt \
# ./output/kinetics200_km_resnet26_3d_v2_0init_tem_reciprocal16_3D_length8_stride8_dropout0.4/model_best.pth \
# --arch km_resnet26_3d_v2_0init_tem_reciprocal16 \
# --mode TSN+3D \
# --batch_size 1 \
# --num_segments 10 \
# --input_size 256 \
# --t_length 8 \
# --t_stride 8 \
# --crop_fusion_type max \
# --dropout 0.4 \
# --workers 16 \
# --image_tmpl image_{:06d}.jpg \
# --save_scores ./output/kinetics200_km_resnet26_3d_v2_0init_tem_reciprocal16_3D_length8_stride8_dropout0.4 \

################################## softmax temperature 1/4 ######################################
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# python main.py \
# kinetics200 \
# data/kinetics200/kinetics200_train_list.txt \
# data/kinetics200/kinetics200_val_list.txt \
# --arch km_resnet26_3d_v2_0init_tem_reciprocal4 \
# --dro 0.4 \
# --mode 3D \
# --t_length 8 \
# --t_stride 8 \
# --epochs 70 \
# --batch-size 128 \
# --lr 0.01 \
# --lr_steps 40 60 \
# --workers 16 \
# --eval-freq 4 \
# --pretrained \
# --pretrained_model models/resnet26.pth \
# --resume output/kinetics200_km_resnet26_3d_v2_0init_tem_reciprocal4_3D_length8_stride8_dropout0.4/checkpoint_52epoch.pth

# python ./test_kaiming.py \
# kinetics200 \
# data/kinetics200/kinetics200_val_list.txt \
# ./output/kinetics200_km_resnet26_3d_v2_0init_tem_reciprocal4_3D_length8_stride8_dropout0.4/model_best.pth \
# --arch km_resnet26_3d_v2_0init_tem_reciprocal4 \
# --mode TSN+3D \
# --batch_size 1 \
# --num_segments 10 \
# --input_size 256 \
# --t_length 8 \
# --t_stride 8 \
# --crop_fusion_type max \
# --dropout 0.4 \
# --workers 16 \
# --image_tmpl image_{:06d}.jpg \
# --save_scores ./output/kinetics200_km_resnet26_3d_v2_0init_tem_reciprocal4_3D_length8_stride8_dropout0.4 \

################################## softmax temperature 1/64 ######################################
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# python main.py \
# kinetics200 \
# data/kinetics200/kinetics200_train_list.txt \
# data/kinetics200/kinetics200_val_list.txt \
# --arch km_resnet26_3d_v2_0init_tem_reciprocal64 \
# --dro 0.4 \
# --mode 3D \
# --t_length 8 \
# --t_stride 8 \
# --epochs 70 \
# --batch-size 128 \
# --lr 0.01 \
# --lr_steps 40 60 \
# --workers 16 \
# --eval-freq 2 \
# --pretrained \
# --pretrained_model models/resnet26.pth \
# --resume output/kinetics200_km_resnet26_3d_v2_0init_tem_reciprocal64_3D_length8_stride8_dropout0.4/checkpoint_26epoch.pth

# python ./test_kaiming.py \
# kinetics200 \
# data/kinetics200/kinetics200_val_list.txt \
# ./output/kinetics200_km_resnet26_3d_v2_0init_tem_reciprocal64_3D_length8_stride8_dropout0.4/model_best.pth \
# --arch km_resnet26_3d_v2_0init_tem_reciprocal64 \
# --mode TSN+3D \
# --batch_size 1 \
# --num_segments 10 \
# --input_size 256 \
# --t_length 8 \
# --t_stride 8 \
# --crop_fusion_type max \
# --dropout 0.4 \
# --workers 16 \
# --image_tmpl image_{:06d}.jpg \
# --save_scores ./output/kinetics200_km_resnet26_3d_v2_0init_tem_reciprocal64_3D_length8_stride8_dropout0.4 \

################################## Temporal Conv vs. TKM-Conv ######################################
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# python main.py \
# kinetics200 \
# data/kinetics200/kinetics200_train_list.txt \
# data/kinetics200/kinetics200_val_list.txt \
# --arch pib_resnet26_3d_full \
# --dro 0.4 \
# --mode 3D \
# --t_length 8 \
# --t_stride 8 \
# --epochs 70 \
# --batch-size 128 \
# --lr 0.01 \
# --lr_steps 40 60 \
# --workers 16 \
# --eval-freq 2 \
# --pretrained \
# --pretrained_model models/resnet26.pth \

# python ./test_kaiming.py \
# kinetics200 \
# data/kinetics200/kinetics200_val_list.txt \
# ./output/kinetics200_pib_resnet26_3d_full_3D_length8_stride8_dropout0.4/model_best.pth \
# --arch pib_resnet26_3d_full \
# --mode TSN+3D \
# --batch_size 1 \
# --num_segments 10 \
# --input_size 256 \
# --t_length 8 \
# --t_stride 8 \
# --crop_fusion_type max \
# --dropout 0.4 \
# --workers 16 \
# --image_tmpl image_{:06d}.jpg \
# --save_scores ./output/kinetics200_pib_resnet26_3d_full_3D_length8_stride8_dropout0.4 \

################################## Pre vs. Post ######################################
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# python main.py \
# kinetics200 \
# data/kinetics200/kinetics200_train_list.txt \
# data/kinetics200/kinetics200_val_list.txt \
# --arch km_resnet26_3d_v2_0init_tem_reciprocal16_pre \
# --dro 0.4 \
# --mode 3D \
# --t_length 8 \
# --t_stride 8 \
# --epochs 70 \
# --batch-size 128 \
# --lr 0.01 \
# --lr_steps 40 60 \
# --workers 8 \
# --eval-freq 2 \
# --pretrained \
# --pretrained_model models/resnet26.pth \
# --resume output/kinetics200_km_resnet26_3d_v2_0init_tem_reciprocal16_pre_3D_length8_stride8_dropout0.4/checkpoint_36epoch.pth

# python ./test_kaiming.py \
# kinetics200 \
# data/kinetics200/kinetics200_val_list.txt \
# ./output/kinetics200_km_resnet26_3d_v2_0init_tem_reciprocal16_pre_3D_length8_stride8_dropout0.4/model_best.pth \
# --arch km_resnet26_3d_v2_0init_tem_reciprocal16_pre \
# --mode TSN+3D \
# --batch_size 1 \
# --num_segments 10 \
# --input_size 256 \
# --t_length 8 \
# --t_stride 8 \
# --crop_fusion_type max \
# --dropout 0.4 \
# --workers 16 \
# --image_tmpl image_{:06d}.jpg \
# --save_scores ./output/kinetics200_km_resnet26_3d_v2_0init_tem_reciprocal16_pre_3D_length8_stride8_dropout0.4 \

################################## softmax temperature 1/10 ######################################
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# python main.py \
# kinetics200 \
# data/kinetics200/kinetics200_train_list.txt \
# data/kinetics200/kinetics200_val_list.txt \
# --arch km_resnet26_3d_v2_0init_tem_reciprocal10 \
# --dro 0.4 \
# --mode 3D \
# --t_length 8 \
# --t_stride 8 \
# --epochs 70 \
# --batch-size 128 \
# --lr 0.01 \
# --lr_steps 40 60 \
# --workers 8 \
# --eval-freq 2 \
# --pretrained \
# --pretrained_model models/resnet26.pth \

# python ./test_kaiming.py \
# kinetics200 \
# data/kinetics200/kinetics200_val_list.txt \
# ./output/kinetics200_km_resnet26_3d_v2_0init_tem_reciprocal10_3D_length8_stride8_dropout0.4/model_best.pth \
# --arch km_resnet26_3d_v2_0init_tem_reciprocal10 \
# --mode TSN+3D \
# --batch_size 1 \
# --num_segments 10 \
# --input_size 256 \
# --t_length 8 \
# --t_stride 8 \
# --crop_fusion_type max \
# --dropout 0.4 \
# --workers 16 \
# --image_tmpl image_{:06d}.jpg \
# --save_scores ./output/kinetics200_km_resnet26_3d_v2_0init_tem_reciprocal10_3D_length8_stride8_dropout0.4 \

################################## softmax temperature 1/20 ######################################
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# python main.py \
# kinetics200 \
# data/kinetics200/kinetics200_train_list.txt \
# data/kinetics200/kinetics200_val_list.txt \
# --arch km_resnet26_3d_v2_0init_tem_reciprocal20 \
# --dro 0.4 \
# --mode 3D \
# --t_length 8 \
# --t_stride 8 \
# --epochs 70 \
# --batch-size 128 \
# --lr 0.01 \
# --lr_steps 40 60 \
# --workers 8 \
# --eval-freq 2 \
# --pretrained \
# --pretrained_model models/resnet26.pth \

# python ./test_kaiming.py \
# kinetics200 \
# data/kinetics200/kinetics200_val_list.txt \
# ./output/kinetics200_km_resnet26_3d_v2_0init_tem_reciprocal20_3D_length8_stride8_dropout0.4/model_best.pth \
# --arch km_resnet26_3d_v2_0init_tem_reciprocal20 \
# --mode TSN+3D \
# --batch_size 1 \
# --num_segments 10 \
# --input_size 256 \
# --t_length 8 \
# --t_stride 8 \
# --crop_fusion_type max \
# --dropout 0.4 \
# --workers 16 \
# --image_tmpl image_{:06d}.jpg \
# --save_scores ./output/kinetics200_km_resnet26_3d_v2_0init_tem_reciprocal20_3D_length8_stride8_dropout0.4 \

################################## mask generation: Identity ######################################
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# python main.py \
# kinetics200 \
# data/kinetics200/kinetics200_train_list.txt \
# data/kinetics200/kinetics200_val_list.txt \
# --arch km_resnet26_3d_v2_1init_prod \
# --dro 0.4 \
# --mode 3D \
# --t_length 8 \
# --t_stride 8 \
# --epochs 70 \
# --batch-size 128 \
# --lr 0.01 \
# --lr_steps 40 60 \
# --workers 16 \
# --eval-freq 2 \
# --pretrained \
# --pretrained_model models/resnet26.pth \

# python ./test_kaiming.py \
# kinetics200 \
# data/kinetics200/kinetics200_val_list.txt \
# ./output/kinetics200_km_resnet26_3d_v2_1init_prod_3D_length8_stride8_dropout0.4/model_best.pth \
# --arch km_resnet26_3d_v2_1init_prod \
# --mode TSN+3D \
# --batch_size 1 \
# --num_segments 10 \
# --input_size 256 \
# --t_length 8 \
# --t_stride 8 \
# --crop_fusion_type max \
# --dropout 0.4 \
# --workers 16 \
# --image_tmpl image_{:06d}.jpg \
# --save_scores ./output/kinetics200_km_resnet26_3d_v2_1init_prod_3D_length8_stride8_dropout0.4 \

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# python main.py \
# kinetics200 \
# data/kinetics200/kinetics200_train_list.txt \
# data/kinetics200/kinetics200_val_list.txt \
# --arch km_resnet26_3d_v2_sigmoid \
# --dro 0.4 \
# --mode 3D \
# --t_length 8 \
# --t_stride 8 \
# --epochs 70 \
# --batch-size 128 \
# --lr 0.01 \
# --lr_steps 40 60 \
# --workers 16 \
# --eval-freq 2 \
# --pretrained \
# --pretrained_model models/resnet26.pth \

# python ./test_kaiming.py \
# kinetics200 \
# data/kinetics200/kinetics200_val_list.txt \
# ./output/kinetics200_km_resnet26_3d_v2_sigmoid_3D_length8_stride8_dropout0.4/model_best.pth \
# --arch km_resnet26_3d_v2_sigmoid \
# --mode TSN+3D \
# --batch_size 1 \
# --num_segments 10 \
# --input_size 256 \
# --t_length 8 \
# --t_stride 8 \
# --crop_fusion_type max \
# --dropout 0.4 \
# --workers 16 \
# --image_tmpl image_{:06d}.jpg \
# --save_scores ./output/kinetics200_km_resnet26_3d_v2_sigmoid_3D_length8_stride8_dropout0.4 \

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# python main.py \
# kinetics200 \
# data/kinetics200/kinetics200_train_list.txt \
# data/kinetics200/kinetics200_val_list.txt \
# --arch pib_resnet26_2d_full \
# --dro 0.4 \
# --mode 3D \
# --t_length 8 \
# --t_stride 8 \
# --epochs 70 \
# --batch-size 128 \
# --lr 0.01 \
# --lr_steps 40 60 \
# --workers 8 \
# --eval-freq 2 \
# --pretrained \
# --pretrained_model models/resnet26.pth \
# --resume output/kinetics200_pib_resnet26_2d_full_3D_length8_stride8_dropout0.4/checkpoint_4epoch.pth

python ./test_kaiming.py \
kinetics200 \
data/kinetics200/kinetics200_val_list.txt \
./output/kinetics200_pib_resnet26_2d_full_3D_length8_stride8_dropout0.4/model_best.pth \
--arch pib_resnet26_2d_full \
--mode TSN+3D \
--batch_size 1 \
--num_segments 10 \
--input_size 256 \
--t_length 8 \
--t_stride 8 \
--crop_fusion_type max \
--dropout 0.4 \
--workers 16 \
--image_tmpl image_{:06d}.jpg \
--save_scores ./output/kinetics200_pib_resnet26_2d_full_3D_length8_stride8_dropout0.4 \