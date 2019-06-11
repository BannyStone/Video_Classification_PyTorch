CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python main.py \
kinetics400 \
data/kinetics400/kinetics_train_list.txt \
data/kinetics400/kinetics_val_list.txt \
--arch resnet50_3d_nopad \
--dro 0.2 \
--mode 3D \
--t_length 20 \
--t_stride 3 \
--pretrained \
--epochs 110 \
--batch-size 64 \
--lr 0.002 \
--lr_steps 60 90 100 \
--workers 16 \

python ./test_kaiming.py \
kinetics400 \
data/kinetics400/kinetics_val_list.txt \
output/kinetics400_resnet50_3d_v3_3D_length16_stride4_dropout0.2/model_best.pth \
--arch resnet50_3d_v3 \
--mode TSN+3D \
--batch_size 1 \
--num_segments 10 \
--input_size 256 \
--t_length 16 \
--t_stride 4 \
--dropout 0.2 \
--workers 12 \
--image_tmpl image_{:06d}.jpg \
--save_scores output/kinetics400_resnet50_3d_v3_3D_length16_stride4_dropout0.2 \