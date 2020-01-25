#!/bin/sh

python3 main.py --dataset_name OT-Vortex-low --batch_size 10 --img_height 256 --img_width 256 --n_cpu 8 --n_epochs 5000 --sample_interval 100 --in_channels 3 --out_channels 3 --dropout_ratio_UNet 0.5 --checkpoint_interval 500
