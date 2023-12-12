import cv2
import mmcv
from mmcv.transforms import Compose
from mmengine.utils import track_iter_progress
import torch

# 加载模型
model = torch.load('D:/Senior/lab/mmdetection/projects/BAANet/checkpoints/best_coco_all_iter_3000_thermal_first.pth')  # 替换为你的模型文件路径

# 打印模型的参数名称
for param in model['state_dict']:
    if param.startswith("fusion_module"):
        print(param,end=",")

# print(model['state_dict'])



