python tools/test_net.py \
    --config-file configs/fcos/fcos_R_101_FPN_2x_person8.yaml \
    MODEL.WEIGHT /data/mnist/data_drive/fcos_R_101_FPN_2x_person8_man_shadows/fcos_R_101_FPN_2x_person8/model_0001000.pth \
    TEST.IMS_PER_BATCH 4