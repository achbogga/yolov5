#!/bin/bash
finetune_weights="/home/aboggaram/models/yolov5x-seg.pt"
dataset_config="/home/aboggaram/projects/yolov5/data/octiva.yaml"
today=$(date +"%Y-%m-%d")
image_size=640
time python3 -m torch.distributed.run \
    --nproc_per_node 2 \
    segment/train.py \
    --device 0,1 \
    --epochs 1000 \
    --batch-size 32 \
    --data "${dataset_config}" \
    --img "${image_size}" \
    --weights "${finetune_weights}" \
    --project "/home/aboggaram/models/octiva_yolov5_instance_segmentation_${today}" \
    --name "train_image_size_${image_size}_" \
	 | tee "/home/aboggaram/logs/octiva_yolov5_${image_size}_instance_segmentation_${today}.log"

