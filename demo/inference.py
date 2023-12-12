from mmdet.apis import init_detector, inference_detector
import mmcv
import torch
import numpy as np
import matplotlib.pyplot as plt

# 指定模型的配置文件和 checkpoint 文件路径
config_file = 'D:/Senior/lab/mmdetection/projects/Distillation/configs/Distill_r50_fpn_1x_kaist_1026.py'
checkpoint_file =r'D:/Senior/lab/mmdetection/projects/Distillation/checkpoints/best_coco_all_iter_10500.pth'

# 根据配置文件和 checkpoint 文件构建模型
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# 测试单张图片并展示结果
img = 'D:/Senior/lab/KAIST/kaist_test_anno/kaist_test/kaist_test_lwir/set06_V001_I00579'  # 或者 img = mmcv.imread(img)，这样图片仅会被读一次
result,mask = inference_detector(model, img)
# 在一个新的窗口中将结果可视化

current_shape =mask.shape
mask_gt = torch.zeros(current_shape,device='cuda:0')

gt_bboxes= torch.tensor([ [
                19,
                213,
                51,
                131
            ],[
                108,
                210,
                46,
                122
            ],[
                65,
                211,
                55,
                124
            ],[
                305,
                225,
                72,
                171
            ],[
                338,
                214,
                35,
                67
            ],[
                2,
                212,
                44,
                83
            ],[
                383,
                212,
                33,
                65
            ]],device = 'cuda:0')

for i in range(len(result)):

    bboxes = gt_bboxes
    bboxes = torch.floor(bboxes*1.5625)         
    print(bboxes)   

    for bbox in bboxes:
        xmin, ymin, width, height = bbox
        mask_gt[i, :, int(ymin):int(ymin+height), int(xmin):int(xmin+width)] = 1

summed_mask_gt = torch.sum(mask_gt, dim=1)  # 在第二个维度上求和，得到形状为[1, 200, 256]的张量

# 移除多余的维度
summed_mask_gt = summed_mask_gt.squeeze()  # 去掉大小为1的维度，得到形状为[200, 256]的张量

# 将PyTorch张量转换为NumPy数组
summed_mask_np_gt = summed_mask_gt.cpu().numpy()

summed_mask = torch.sum(mask, dim=1)  # 在第二个维度上求和，得到形状为[1, 200, 256]的张量

# 移除多余的维度
summed_mask = summed_mask.squeeze()  # 去掉大小为1的维度，得到形状为[200, 256]的张量

# 将PyTorch张量转换为NumPy数组
summed_mask_np = summed_mask.cpu().numpy()
print(summed_mask_np.max())
# 可视化融合后的遮罩
plt.subplot(1, 2, 1)
plt.imshow(summed_mask_np, cmap='gray')
plt.subplot(1, 2, 2)
plt.imshow(summed_mask_np_gt, cmap='gray')
plt.show()
print(mask_gt.max())

