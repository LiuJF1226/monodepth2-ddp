# Monodepth2-DDP

This is a personal modified PyTorch implementation (not official) for [Monodepth2](https://github.com/nianticlabs/monodepth2) ("Digging into Self-Supervised Monocular Depth Prediction", ICCV2019).


On the basis of the raw code in [Monodepth2](https://github.com/nianticlabs/monodepth2), we add some new features in this version:
* DDP training mode
* Resume from an interrupted training automatically
* Evaluate and log after each epoch
* NYUv2 training and evaluation
* Make3D evaluation

Todo:
* Cityscapes training and evaluation.


# Setup

Install the dependencies with:
```shell
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html

pip install scikit-image timm thop yacs opencv-python h5py
```
We experiment with PyTorch 1.9.1, CUDA 11.1, Python 3.7. Other torch versions may also be okay.


# Preparing datasets

For KITTI and KITTI Odometry datasets, you can prepare them as done in [Monodepth2](https://github.com/nianticlabs/monodepth2). Note that we directly train with the raw png images and do not convert them to jpgs. You also need to generate the groundtruth depth maps before training since the code will evaluate after each epoch. For the  groundtruth, run the following commands:
```shell
### generate raw groundtruth
python export_gt_depth.py --data_path /home/datasets/kitti_raw_data --split eigen

### generate improved groundtruth
python export_gt_depth.py --data_path /home/datasets/kitti_raw_data --split eigen_benchmark
```

For NYUv2 dataset, you can download the training and testing datasets as done in [StructDepth](https://github.com/SJTU-ViSYS/StructDepth).

For Make3D dataset, you can download it from [here](http://make3d.cs.cornell.edu/data.html#make3d).


# Training
You can see the training scripts in [run_kitti.sh](./run_kitti.sh) and [run_nyu.sh](./run_nyu.sh). Take the KITTI script as an example:
```shell
# CUDA_VISIBLE_DEVICES=0 python train.py \

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py \
--data_path /home/datasets/kitti_raw_data \
--dataset kitti \
--log_dir /home/jinfengliu/logs \
--exp_name mono_kitti \
--width 640 \
--height 192 \
--num_scales 4 \
--batch_size 12 \
--lr_sche_type step \
--learning_rate 1e-4 \
--eta_min 5e-6 \
--num_epochs 20 \
--decay_step 15 \
--decay_rate 0.1 \
--log_frequency 400 \
--save_frequency 400 \
--resume 
# --pretrained_path xxxx/ckpt.pth
```
This is a monocular training example on KITTI. If you want to conduct stereo training or monocular+stereo training, please refer to [Monodepth2](https://github.com/nianticlabs/monodepth2) to specify `--frame_ids` and `--use_stereo` flags.

Use `--split` flag to specify the training split on KITTI (see [Monodepth2](https://github.com/nianticlabs/monodepth2)), and default is eigen_zhou.

Use `CUDA_VISIBLE_DEVICES=0 python train.py` to train with a single GPU. If you want to train with two or more GPUs, then use `CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py` for DDP training.

Use `--data_path` flag to specify the dataset folder.

Use `--log_dir` flag to specify the logging folder.

Use `--exp_name` flag to specify the experiment name.

All output files (checkpoints, logs and tensorboard) will be saved in the directory `{log_dir}/{exp_name}`.

Use `--pretrained_path` flag to load a pretrained checkpoint if necessary.

Look at [options.py](./options.py) to see the range of other training options.

# Evaluation
You can see the evaluation scripts in [evaluate.sh](./evaluate.sh). 

## Depth evaluation
```shell
CUDA_VISIBLE_DEVICES=0 python evaluate_depth.py \
--pretrained_path /home/jinfengliu/logs/mono2_official/mono_640x192.pth \
--batch_size 12 \
--kitti_path /home/datasets/kitti_raw_data \
--make3d_path /home/datasets/make3d \
--nyuv2_path /home/datasets/nyu_v2 \
# --use_stereo
# --post_process
```
This script will evaluate on KITTI (both raw and improved GT), NYUv2 and Make3D together. If you don't want to evaluate on one of these datasets, for example KITTI, just do not specify the corresponding `--kitti_path` flag. It will only evaluate on the datasets which you have specified a path flag.

If the model is under stereo training, add the `--use_stereo` flag.

If you want to evalute with post-processing, add the `--post_process` flag.

## Pose evaluation on KITTI Odometry
```shell
CUDA_VISIBLE_DEVICES=0 python evaluate_pose.py \
--pretrained_path /home/jinfengliu/logs/mono2_official/mono_640x192.pth \
--data_path /home/datasets/kitti_odometry \
--batch_size 12 \
--eval_split odom_9 
```
The `--eval_split` flag can only be odom_9 or odom_10.


# Prediction

## Prediction for a single image
You can predict scaled disparity for a single image with:

```shell
python test_simple.py --image_path folder/test_image.jpg --pretrained_path xxxx/ckpt.pth --save_npy
```

or, if you are using a stereo-trained model, you can estimate metric depth with:

```shell
python test_simple.py --image_path folder/test_image.jpg --pretrained_path xxxx/ckpt.pth --save_npy --pred_metric_depth
```

The `--image_path` flag can also be a directory containing several images. In this setting, the script will predict all the images (use `--ext` to specify png or jpg) in the directory:

```shell
python test_simple.py --image_path folder --pretrained_path xxxx/ckpt.pth --ext png --save_npy
```

## Prediction for a video

```shell
python test_video.py --image_path folder --pretrained_path xxxx/ckpt.pth --ext png
```
Here the `--image_path` flag should be a directory containing several video frames. Note that these video frame files should be named in an ascending numerical order. For example, the first frame is named as `0000.png`, the second frame is named as `0001.png`, and etc. Then the script will output a GIF file.

# Acknowledgement
We have used codes from other wonderful open-source projects,
[Monodepth2](https://github.com/nianticlabs/monodepth2) and [StructDepth](https://github.com/SJTU-ViSYS/StructDepth). Thanks for their excellent works!