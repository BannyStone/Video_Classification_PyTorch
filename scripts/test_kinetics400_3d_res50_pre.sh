CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python ./test.py \
kinetics400 \
data/kinetics400/kinetics_val_list.txt \
./output/kinetics400_resnet50_3d_3D_length16_stride4_dropout0.2/model_best.pth \
--arch resnet50_3d \
--mode TSN+3D \
--batch_size 2 \
--num_segments 15 \
--test_crops 10 \
--t_length 16 \
--t_stride 4 \
--crop_fusion_type max \
--dropout 0.2 \
--workers 12 \
--save_scores ./output/kinetics400_resnet50_3d_3D_length16_stride4_dropout0.2