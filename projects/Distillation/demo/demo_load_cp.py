import cv2
import mmcv
from mmcv.transforms import Compose
from mmengine.utils import track_iter_progress
from mmdet.registry import VISUALIZERS
from ..distillation.api import init_backbone
import torch 

config_file = './projects/BAANet/configs/BAANet_r50_fpn_1x_kaist.py'
checkpoint_file = './projects/BAANet/checkpoints/best_coco_bbox_mAP_epoch_9.pth'

# 根据配置文件和 checkpoint 文件构建模型
model = init_backbone(config_file, checkpoint_file, device='cuda:0')

img = 'D:/Senior/lab/KAIST/kaist_test_anno/kaist_test/kaist_test_lwir/set06_V001_I00519_lwir.png'  # 或者 img = mmcv.imread(img)，这样图片仅会被读一次


inputs = torch.randn(1,6,512,640)
device ='cuda:0'
inputs = inputs.to(device)
result = model(inputs)
print(len(result))
for i in range(len(result)):
    print(result[i].shape)