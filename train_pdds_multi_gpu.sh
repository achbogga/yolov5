#!/bin/bash
finetune_weights="/home/aboggaram/models/pdds/yolov5x.pt"
dataset_config="/home/aboggaram/data/spinach_consolidated_yolov5_Dec_20_2022/yolov5_dataset.yaml"
hyperparameter_config="/home/aboggaram/projects/yolov5/data/hyps/pdds_hyp.scratch-low.yaml"
today=$(date +"%Y-%m-%d")
image_size=640
time python3 -m torch.distributed.run \
    --nproc_per_node 2 \
    train.py \
    --device 0,1 \
    --epochs 500 \
    --hyp "${hyperparameter_config}" \
    --optimizer "SGD" \
    --batch-size 16 \
    --multi-scale \
    --data "${dataset_config}" \
    --img "${image_size}" \
    --weights "${finetune_weights}" \
    --project "/home/aboggaram/models/pdds/spinach_pdds_${today}" \
    --name "train_image_size_${image_size}_" \
	 | tee "/home/aboggaram/logs/pdds_yolov5_${image_size}_spinach_${today}.log"

