# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# python main.py \
# kinetics400 \
# data/kinetics400/kinetics_train_list.txt \
# data/kinetics400/kinetics_val_list.txt \
# --arch pib_resnet50_3d_slow \
# --dro 0.4 \
# --mode 3D \
# --t_length 8 \
# --t_stride 8 \
# --epochs 110 \
# --batch-size 96 \
# --lr 0.01 \
# --wd 0.0001 \
# --lr_steps 40 70 90 \
# --workers 16 \
# --pretrained \
# --resume output/kinetics400_pib_resnet50_3d_slow_3D_length8_stride8_dropout0.4/history_best/model_best.pth

# python ./test_kaiming.py \
# kinetics400 \
# data/kinetics400/kinetics_val_list.txt \
# ./output/kinetics400_pib_resnet50_3d_slow_3D_length8_stride8_dropout0.4/history_best/model_best.pth \
# --arch pib_resnet50_3d_slow \
# --mode TSN+3D \
# --batch_size 1 \
# --num_segments 10 \
# --input_size 256 \
# --t_length 8 \
# --t_stride 8 \
# --crop_fusion_type avg \
# --dropout 0.2 \
# --workers 10 \
# --image_tmpl image_{:06d}.jpg \
# --save_scores ./output/kinetics400_pib_resnet50_3d_slow_3D_length8_stride8_dropout0.4 \

# CUDA_LAUNCH_BLOCKING=1 \
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# python main.py \
# kinetics400 \
# data/kinetics400/kinetics_train_list.txt \
# data/kinetics400/kinetics_val_list.txt \
# --arch pib_resnet50_3d_slow \
# --dro 0.2 \
# --mode 3D \
# --t_length 8 \
# --t_stride 8 \
# --epochs 110 \
# --batch-size 96 \
# --lr 0.002 \
# --lr_steps 60 90 100 \
# --workers 32 \
# --pretrained \
# --resume output/kinetics400_pib_resnet50_3d_slow_3D_length8_stride8_dropout0.2/checkpoint_32epoch.pth

python ./test_kaiming.py \
kinetics400 \
data/kinetics400/kinetics_val_list.txt \
./output/kinetics400_pib_resnet50_3d_slow_3D_length8_stride8_dropout0.2/model_best.pth \
--arch pib_resnet50_3d_slow \
--mode TSN+3D \
--batch_size 1 \
--num_segments 10 \
--input_size 256 \
--t_length 8 \
--t_stride 8 \
--dropout 0.2 \
--workers 10 \
--image_tmpl image_{:06d}.jpg \
