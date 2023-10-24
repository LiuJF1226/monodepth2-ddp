CUDA_VISIBLE_DEVICES=0 python evaluate_depth.py \
--pretrained_path /home/jinfengliu/logs/mono2_official/mono_640x192.pth \
--batch_size 12 \
--num_layers 18 \
--kitti_path /home/datasets/kitti_raw_data \
--make3d_path /home/datasets/make3d \
--cityscapes_path /home/datasets/cityscapes \
--nyuv2_path /home/datasets/nyu_v2 \
# --use_stereo \
# --post_process \


CUDA_VISIBLE_DEVICES=0 python evaluate_pose.py \
--pretrained_path /home/jinfengliu/logs/mono2_official/mono_640x192.pth \
--data_path /home/datasets/kitti_odometry \
--batch_size 12 \
--num_layers 18 \
--eval_split odom_9 \
# --jpg \

