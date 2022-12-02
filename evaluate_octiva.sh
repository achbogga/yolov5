#!/bin/bash
finetune_weights="/home/aboggaram/models/octiva_yolov5_instance_segmentation_2022-11-22/exp/weights/best.pt"
dataset_config="/home/aboggaram/projects/yolov7_instance_segmentation/seg/data/octiva_old.yaml"
today=$(date +"%Y-%m-%d")
time python3 \
    segment/val.py \
    --data "${dataset_config}" \
    --batch-size 16 \
    --img 640 \
    --conf-thres 0.15 \
    --weights "${finetune_weights}" \
    --name "octiva_yolov5_instance_segmentation_eval_${today}" \
	 | tee "/home/aboggaram/logs/eval_octiva_yolov5_instance_segmentation_${today}.log"

