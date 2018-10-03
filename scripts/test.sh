python ./test.py \
kinetics400 \
data/kinetics400/kinetics_val_list.txt \
./output/kinetics400_resnet50_3d_lite_3D_length8_stride8_dropout0.2/model_best.pth \
--arch resnet50_3d_lite \
--mode TSN+3D \
--batch_size 2 \
--num_segments 20 \
--test_crops 10 \
--t_length 8 \
--t_stride 8 \
--crop_fusion_type avg \
--dropout 0.2 \
--workers 32 \
--save_scores ./output/kinetics400_resnet50_3d_lite_3D_length8_stride8_dropout0.2