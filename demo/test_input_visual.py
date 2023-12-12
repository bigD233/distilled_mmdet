import matplotlib.pyplot as plt
import torch
import matplotlib.patches as patches
tensor = torch.load('D:\Senior\lab\mmdetection\saved_tensor.pt')

# 获取批次大小和通道数
batch_size, num_channels, height, width = tensor.shape

print(num_channels)
# 逐层可视化

bbox_tensor = torch.tensor([
    [244.1250, 297.9375, 275.6250, 375.3750],
    [80.0625, 291.3750, 111.5625, 368.8125],
    [200.8125, 290.0625, 229.6875, 364.8750],
    [161.4375, 283.5000, 192.9375, 358.3125],
    [270.3750, 322.8750, 295.3125, 383.2500]
])
bbox_list = bbox_tensor.cpu().int().tolist()

fig, ax = plt.subplots(1)
three_layer = tensor[0, :3, :, :].cpu()
image_array = three_layer.numpy().transpose(1, 2, 0)

ax.imshow(image_array)
for box in bbox_list:
    x, y, w, h = box[0], box[1], box[2] - box[0], box[3] - box[1]
    rect = patches.Rectangle((x, y), w, h, linewidth=3, edgecolor='r', facecolor='none')
    ax.add_patch(rect)




# 使用 matplotlib 进行可视化
plt.imshow(image_array, cmap='gray')  # 假设是灰度图
plt.title(f'Batch 1, Channel 3-6')
plt.show()