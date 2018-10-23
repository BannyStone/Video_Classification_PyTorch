python ./test.py \
ucf101 \
data/ucf101/ucf101_val_split1_list.txt \
./output/ucf101_resnet50_3d_3D_length16_stride4_dropout0.4_2dpretrained/model_best.pth \
--arch resnet50_3d \
--mode TSN+3D \
--batch_size 1 \
--num_segments 20 \
--test_crops 10 \
--t_length 16 \
--t_stride 4 \
--crop_fusion_type avg \
--dropout 0.4 \
--workers 32 \
--save_scores ./output/ucf101_resnet50_3d_3D_length16_stride4_dropout0.4_2dpretrained \
--image_tmpl img_{:05d}.jpg
