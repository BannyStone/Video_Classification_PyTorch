python ./test.py \
kinetics400 \
data/kinetics400/kinetics_val_list.txt \
./output/kinetics400_resnet50_3d_v1_3D_length16_stride4_dropout0.2/model_best.pth \
--arch resnet50_3d_v1 \
--mode TSN+3D \
--batch_size 2 \
--num_segments 10 \
--test_crops 10 \
--t_length 16 \
--t_stride 4 \
--crop_fusion_type avg \
--dropout 0.2 \
--workers 16 \
--image_tmpl image_{:06d}.jpg \
--save_scores ./output/kinetics400_resnet50_3d_v1_3D_length16_stride4_dropout0.2
