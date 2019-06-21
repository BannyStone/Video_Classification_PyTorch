###################### Baseline: ResNet50 2D ###########################
### Result 1: 17.62+43.69 {dp:0.4,lr:0.01,wd:1e-4,s:[40,50,60],b:96,s:8*4} ###
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python main_20bn.py \
sthsth_v1 \
data/sthsth_v1/sthv1_train_list.txt \
data/sthsth_v1/sthv1_val_list.txt \
--arch resnet50_2d \
--dro 0.4 \
--mode 3D \
--t_length 8 \
--t_stride 4 \
--epochs 40 \
--batch-size 96 \
--lr 0.01 \
--wd 0.0001 \
--lr_steps 20 30 \
--workers 16 \
--image_tmpl "{:05d}.jpg" \
--pretrained \
-ef 1 \
--resume output/sthsth_v1_resnet50_2d_3D_length8_stride4_dropout0.4/checkpoint_20epoch.pth

# python ./test_kaiming.py \
# sthsth_v1 \
# data/sthsth_v1/sthv1_val_list.txt \
# ./output/sthsth_v1_resnet50_2d_3D_length8_stride4_dropout0.4/model_best.pth \
# --arch resnet50_2d \
# --mode TSN+3D \
# --batch_size 1 \
# --num_segments 3 \
# --input_size 256 \
# --t_length 8 \
# --t_stride 4 \
# --dropout 0.4 \
# --workers 16 \
# --image_tmpl {:05d}.jpg \

# CUDA_LAUNCH_BLOCKING=1 \
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# python main_20bn.py \
# sthsth_v1 \
# data/sthsth_v1/sthv1_train_list.txt \
# data/sthsth_v1/sthv1_val_list.txt \
# --arch traj_resnet50_2d_res3 \
# --dro 0.4 \
# --mode 3D \
# --t_length 8 \
# --t_stride 4 \
# --epochs 60 \
# --batch-size 96 \
# --lr 0.01 \
# --wd 0.0001 \
# --lr_steps 40 50 \
# --workers 16 \
# --image_tmpl "{:05d}.jpg" \
# --pretrained \

# python ./test_kaiming.py \
# sthsth_v1 \
# data/sthsth_v1/sthv1_val_list.txt \
# ./output/sthsth_v1_pib_resnet50_3d_slow_3D_length8_stride8_dropout0.4/model_best.pth \
# --arch pib_resnet50_3d_slow \
# --mode TSN+3D \
# --batch_size 1 \
# --num_segments 4 \
# --input_size 256 \
# --t_length 8 \
# --t_stride 8 \
# --crop_fusion_type avg \
# --dropout 0.4 \
# --workers 16 \
# --image_tmpl {:05d}.jpg \
# --save_scores ./output/sthsth_v1_pib_resnet50_3d_slow_3D_length8_stride8_dropout0.4 \
