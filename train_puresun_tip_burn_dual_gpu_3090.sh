python -m torch.distributed.launch --nproc_per_node 2 train.py --batch 64 --data data/yolov5_puresun_tip_burn_dataset.yaml --weights yolov5s.pt --device 0,1
