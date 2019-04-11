# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# python finetune_new.py \
# hmdb51_s1 \
# data/hmdb51/hmdb51_train_split1_list.txt \
# data/hmdb51/hmdb51_val_split1_list.txt \
# --arch gsv_resnet50_3d_v3 \
# --dro 0.2 \
# --mode 3D \
# --t_length 16 \
# --t_stride 4 \
# --pretrained \
# --pretrained_model models/gsv_resnet50_v3.pth \
# --epochs 40 \
# --batch-size 64 \
# --lr 0.0001 \
# --lr_steps 24 \
# --workers 20 \
# --image_tmpl img_{:05d}.jpg \
# -ef 2 \
# --resume output/hmdb51_s1_gsv_resnet50_3d_v3_3D_length16_stride4_dropout0.0/checkpoint_24epoch.pth \

python ./test.py \
hmdb51 \
data/hmdb51/hmdb51_val_split1_list.txt \
./output/hmdb51_gsv_resnet50_3d_v3_3D_length16_stride4_dropout0.4/s1/model_best.pth \
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
--save_scores ./output/hmdb51_gsv_resnet50_3d_v3_3D_length16_stride4_dropout0.4/s1 \

python ./test.py \
hmdb51 \
data/hmdb51/hmdb51_val_split2_list.txt \
./output/hmdb51_s2_gsv_resnet50_3d_v3_3D_length16_stride4_dropout0.4/model_best.pth \
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
--save_scores ./output/hmdb51_s2_gsv_resnet50_3d_v3_3D_length16_stride4_dropout0.4 \

python ./test.py \
hmdb51 \
data/hmdb51/hmdb51_val_split3_list.txt \
./output/hmdb51_s3_gsv_resnet50_3d_v3_3D_length16_stride4_dropout0.4/model_best.pth \
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
--save_scores ./output/hmdb51_s3_gsv_resnet50_3d_v3_3D_length16_stride4_dropout0.4 \

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# python main.py \
# hmdb51_s2 \
# data/hmdb51/hmdb51_train_split2_list.txt \
# data/hmdb51/hmdb51_val_split2_list.txt \
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

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# python main.py \
# hmdb51_s3 \
# data/hmdb51/hmdb51_train_split3_list.txt \
# data/hmdb51/hmdb51_val_split3_list.txt \
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
