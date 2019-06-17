# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# python main.py \
# kinetics400 \
# data/kinetics400/kinetics_train_list_xlw \
# data/kinetics400/kinetics_val_list_xlw \
# --arch resnet50_3d_nodown \
# --dro 0.2 \
# --mode 3D \
# --t_length 8 \
# --t_stride 8 \
# --pretrained \
# --epochs 110 \
# --batch-size 96 \
# --lr 0.002 \
# --lr_steps 60 90 100 \
# --workers 16 \

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python main.py \
kinetics400 \
data/kinetics400/kinetics_train_list_xlw \
data/kinetics400/kinetics_val_list_xlw \
--arch resnet50_3d_nodown \
--dro 0.5 \
--mode 3D \
--t_length 8 \
--t_stride 8 \
--pretrained \
--epochs 110 \
--batch-size 96 \
--lr 0.02 \
--wd 0.0001 \
--lr_steps 40 80 100 \
--workers 16 \

# CUDA_LAUNCH_BLOCKING=1 \
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# python main.py \
# kinetics400 \
# data/kinetics400/kinetics_train_list.txt \
# data/kinetics400/kinetics_val_list.txt \
# --arch resnet50_3d_nodown \
# --dro 0.2 \
# --mode 3D \
# --t_length 8 \
# --t_stride 8 \
# --pretrained \
# --epochs 110 \
# --batch-size 96 \
# --lr 0.002 \
# --lr_steps 60 90 100 \
# --workers 16 \
# --resume output/kinetics400_resnet50_3d_nodown_3D_length8_stride8_dropout0.2/checkpoint_82epoch.pth

# python ./test_kaiming.py \
# kinetics400 \
# data/kinetics400/kinetics_val_list_xlw \
# output/kinetics400_resnet50_3d_nodown_3D_length8_stride8_dropout0.2/model_best.pth \
# --arch resnet50_3d_nodown \
# --mode TSN+3D \
# --batch_size 1 \
# --num_segments 10 \
# --input_size 256 \
# --t_length 8 \
# --t_stride 8 \
# --dropout 0.2 \
# --workers 12 \
# --image_tmpl image_{:06d}.jpg \

# python ./test.py \
# kinetics400 \
# data/kinetics400/kinetics_val_list.txt \
# output/kinetics400_resnet50_3d_nodown_3D_length8_stride8_dropout0.2/model_best.pth \
# --arch resnet50_3d_nodown \
# --mode TSN+3D \
# --batch_size 1 \
# --num_segments 10 \
# --input_size 224 \
# --t_length 8 \
# --t_stride 8 \
# --dropout 0.2 \
# --workers 12 \
# --image_tmpl image_{:06d}.jpg \
