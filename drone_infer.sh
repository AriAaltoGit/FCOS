python tools/test_net.py \
    --config-file fcos/configs/fcos_R_101_FPN_2x.yaml \
    MODEL.WEIGHT trained_models/fcos_R_101_FPN_2x_person8/model_final.pth \
    TEST.IMS_PER_BATCH 1