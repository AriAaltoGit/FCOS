python -m torch.distributed.launch \
    --nproc_per_node=1 \
    --nnodes=1 \
    --node_rank=0 \
    --master_port='12340' \
    --master_addr=localhost \
    tools/train_net.py \
    --config-file fcos/configs/fcos_R_101_FPN_2x.yaml 
    
