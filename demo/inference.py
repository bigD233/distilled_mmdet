from mmdet.apis import init_detector, inference_detector
import mmcv

# 指定模型的配置文件和 checkpoint 文件路径
config_file = './projects\BAANet\configs\BAANet_r50_fpn_1x_kaist.py'
checkpoint_file = './projects/BAANet/checkpoints/best_coco_bbox_mAP_epoch_9.pth'

# 根据配置文件和 checkpoint 文件构建模型
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# 测试单张图片并展示结果
img = 'D:\大四\实验室\KAIST\kaist_test_anno\kaist_test\kaist_test_lwir\set06_V001_I00579_lwir.png'  # 或者 img = mmcv.imread(img)，这样图片仅会被读一次
result = inference_detector(model, img)
# 在一个新的窗口中将结果可视化

print(result)

