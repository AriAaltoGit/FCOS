import torch
path = '/data/mnist/data_drive/fcos_trained_models/fcos_R_101_FPN_2x_person8_man_shadows/inference/voc_man_shadows_frame_test/predictions.pth'
model = torch.load(path)
print(model[0].bbox)