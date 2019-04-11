# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# python main.py \
# ucf101_s1 \
# data/ucf101/ucf101_train_split1_list.txt \
# data/ucf101/ucf101_val_split1_list.txt \
# --arch gsv_resnet50_3d_v3 \
# --dro 0.4 \
# --mode 3D \
# --t_length 16 \
# --t_stride 4 \
# --pretrained \
# --pretrained_model models/gsv_resnet50_v3.pth \
# --epochs 30 \
# --batch-size 64 \
# --lr 0.0001 \
# --lr_steps 20 \
# --workers 20 \
# --image_tmpl img_{:05d}.jpg \
# -ef 2 \

python ./test.py \
ucf101 \
data/ucf101/ucf101_val_split1_list.txt \
./output/ucf101_s1_gsv_resnet50_3d_v3_3D_length16_stride4_dropout0.4/model_best.pth \
--arch gsv_resnet50_3d_v3 \
--mode TSN+3D \
--batch_size 2 \
--num_segments 10 \
--t_length 16 \
--t_stride 4 \
--crop_fusion_type max \
--dropout 0.4 \
--workers 8 \
--image_tmpl img_{:05d}.jpg \
--save_scores ./output/ucf101_s1_gsv_resnet50_3d_v3_3D_length16_stride4_dropout0.4 \

# CUDA_LAUNCH_BLOCKING=1 \
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# python main.py \
# ucf101 \
# data/ucf101/ucf101_train_split2_list.txt \
# data/ucf101/ucf101_val_split2_list.txt \
# --arch gsv_resnet50_3d_v3 \
# --dro 0.4 \
# --mode 3D \
# --t_length 16 \
# --t_stride 4 \
# --pretrained \
# --pretrained_model models/gsv_resnet50_v3.pth \
# --epochs 10 \
# --batch-size 64 \
# --lr 0.001 \
# --lr_steps 4 8 \
# --workers 16 \
# --image_tmpl img_{:05d}.jpg \
# -ef 1 \
# --resume output/ucf101_gsv_resnet50_3d_v3_3D_length16_stride4_dropout0.4/checkpoint_4epoch.pth \

# python ./test.py \
# ucf101 \
# data/ucf101/ucf101_val_split2_list.txt \
# ./output/ucf101_gsv_resnet50_3d_v3_3D_length16_stride4_dropout0.4/checkpoint_8epoch.pth \
# --arch gsv_resnet50_3d_v3 \
# --mode TSN+3D \
# --batch_size 4 \
# --num_segments 6 \
# --t_length 16 \
# --t_stride 4 \
# --crop_fusion_type max \
# --dropout 0.4 \
# --workers 16 \
# --image_tmpl img_{:05d}.jpg \
# --save_scores ./output/ucf101_gsv_resnet50_3d_v3_3D_length16_stride4_dropout0.4 \

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python main.py \
ucf101 \
data/ucf101/ucf101_train_split3_list.txt \
data/ucf101/ucf101_val_split3_list.txt \
--arch gsv_resnet50_3d_v3 \
--dro 0.4 \
--mode 3D \
--t_length 16 \
--t_stride 4 \
--pretrained \
--pretrained_model models/gsv_resnet50_v3.pth \
--epochs 14 \
--batch-size 64 \
--lr 0.001 \
--lr_steps 6 12 \
--workers 16 \
--image_tmpl img_{:05d}.jpg \
-ef 1 \
--resume output/ucf101_gsv_resnet50_3d_v3_3D_length16_stride4_dropout0.4/checkpoint_6epoch.pth \

python ./test.py \
ucf101 \
data/ucf101/ucf101_val_split3_list.txt \
./output/ucf101_gsv_resnet50_3d_v3_3D_length16_stride4_dropout0.4/model_best.pth \
--arch gsv_resnet50_3d_v3 \
--mode TSN+3D \
--batch_size 2 \
--num_segments 10 \
--t_length 16 \
--t_stride 4 \
--crop_fusion_type max \
--dropout 0.4 \
--workers 10 \
--image_tmpl img_{:05d}.jpg \
--save_scores ./output/ucf101_gsv_resnet50_3d_v3_3D_length16_stride4_dropout0.4 \