# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# python main.py \
# ucf101_s1 \
# data/ucf101/ucf101_train_split1_list.txt \
# data/ucf101/ucf101_val_split1_list.txt \
# --arch km_resnet50_3d_v2_0init_tem_reciprocal16 \
# --dro 0.4 \
# --mode 3D \
# --t_length 16 \
# --t_stride 4 \
# --pretrained \
# --pretrained_model models/km_res50_r16.pth \
# --epochs 40 \
# --batch-size 64 \
# --lr 0.0001 \
# --lr_steps 30 \
# --workers 16 \
# --image_tmpl img_{:05d}.jpg \
# -ef 2 \
# --resume output/ucf101_s1_km_resnet50_3d_v2_0init_tem_reciprocal16_3D_length16_stride4_dropout0.4/checkpoint_30epoch.pth

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python finetune_new.py \
ucf101_s1 \
data/ucf101/ucf101_train_split1_list.txt \
data/ucf101/ucf101_val_split1_list.txt \
--arch km_resnet50_3d_v2_0init_tem_reciprocal16 \
--dro 0.8 \
--mode 3D \
--t_length 16 \
--t_stride 4 \
--pretrained \
--pretrained_model models/km_res50_r16.pth \
--epochs 40 \
--batch-size 64 \
--lr 0.001 \
--lr_steps 20 30 \
--workers 16 \
--image_tmpl img_{:05d}.jpg \
-ef 2 \

# python ./test_kaiming.py \
# ucf101 \
# data/ucf101/ucf101_val_split1_list.txt \
# ./output/ucf101_s1_km_resnet50_3d_v2_0init_tem_reciprocal16_3D_length16_stride4_dropout0.4/model_best.pth \
# --arch km_resnet50_3d_v2_0init_tem_reciprocal16 \
# --mode TSN+3D \
# --batch_size 1 \
# --num_segments 10 \
# --input_size 256 \
# --t_length 16 \
# --t_stride 4 \
# --crop_fusion_type max \
# --dropout 0.4 \
# --workers 16 \
# --image_tmpl img_{:05d}.jpg \
# --save_scores ./output/ucf101_s1_km_resnet50_3d_v2_0init_tem_reciprocal16_3D_length16_stride4_dropout0.4 \

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# python main.py \
# ucf101_s2 \
# data/ucf101/ucf101_train_split2_list.txt \
# data/ucf101/ucf101_val_split2_list.txt \
# --arch km_resnet50_3d_v2_0init_tem_reciprocal16 \
# --dro 0.4 \
# --mode 3D \
# --t_length 16 \
# --t_stride 4 \
# --pretrained \
# --pretrained_model models/km_res50_r16.pth \
# --epochs 30 \
# --batch-size 64 \
# --lr 0.0001 \
# --lr_steps 20 \
# --workers 16 \
# --image_tmpl img_{:05d}.jpg \
# -ef 2 \
# --resume output/ucf101_s2_km_resnet50_3d_v2_0init_tem_reciprocal16_3D_length16_stride4_dropout0.4/checkpoint_20epoch.pth

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# python main.py \
# ucf101_s3 \
# data/ucf101/ucf101_train_split3_list.txt \
# data/ucf101/ucf101_val_split3_list.txt \
# --arch km_resnet50_3d_v2_0init_tem_reciprocal16 \
# --dro 0.4 \
# --mode 3D \
# --t_length 16 \
# --t_stride 4 \
# --pretrained \
# --pretrained_model models/km_res50_r16.pth \
# --epochs 40 \
# --batch-size 64 \
# --lr 0.0001 \
# --lr_steps 30 \
# --workers 16 \
# --image_tmpl img_{:05d}.jpg \
# -ef 2 \
# --resume output/ucf101_s3_km_resnet50_3d_v2_0init_tem_reciprocal16_3D_length16_stride4_dropout0.4/checkpoint_20epoch.pth

# python ./test_kaiming.py \
# ucf101 \
# data/ucf101/ucf101_val_split2_list.txt \
# ./output/ucf101_s2_km_resnet50_3d_v2_0init_tem_reciprocal16_3D_length16_stride4_dropout0.4/model_best.pth \
# --arch km_resnet50_3d_v2_0init_tem_reciprocal16 \
# --mode TSN+3D \
# --batch_size 1 \
# --num_segments 10 \
# --input_size 256 \
# --t_length 16 \
# --t_stride 4 \
# --crop_fusion_type max \
# --dropout 0.4 \
# --workers 16 \
# --image_tmpl img_{:05d}.jpg \
# --save_scores ./output/ucf101_s2_km_resnet50_3d_v2_0init_tem_reciprocal16_3D_length16_stride4_dropout0.4 \

# python ./test_kaiming.py \
# ucf101 \
# data/ucf101/ucf101_val_split3_list.txt \
# ./output/ucf101_s3_km_resnet50_3d_v2_0init_tem_reciprocal16_3D_length16_stride4_dropout0.4/model_best.pth \
# --arch km_resnet50_3d_v2_0init_tem_reciprocal16 \
# --mode TSN+3D \
# --batch_size 1 \
# --num_segments 10 \
# --input_size 256 \
# --t_length 16 \
# --t_stride 4 \
# --crop_fusion_type max \
# --dropout 0.4 \
# --workers 16 \
# --image_tmpl img_{:05d}.jpg \
# --save_scores ./output/ucf101_s3_km_resnet50_3d_v2_0init_tem_reciprocal16_3D_length16_stride4_dropout0.4 \