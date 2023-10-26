import cv2
import mmcv
from mmcv.transforms import Compose
from mmengine.utils import track_iter_progress
from mmdet.registry import VISUALIZERS
from mmdet.apis import init_backbone,inference_detector
import torch

# 指定模型的配置文件和 checkpoint 文件路径
config_file = './projects/BAANet/configs/BAANet_r50_fpn_1x_kaist.py'

config_file = './projects/Distillation/configs/Distill_r50_fpn_1x_kaist.py'
checkpoint_file = './projects/BAANet/checkpoints/best_coco_bbox_mAP_epoch_9.pth'

# 根据配置文件和 checkpoint 文件构建模型
model = init_backbone(config_file, checkpoint_file, device='cuda:0')

a = torch.randn(1,6,512,640)

device = 'cuda:0'

a= a.to(device)
img = 'test.jpg'  # 或者 img = mmcv.imread(img)，这样图片仅会被读一次
result = model(a)


print(result)
# 显示结果




