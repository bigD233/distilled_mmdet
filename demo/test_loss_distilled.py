import torch 
import torch.nn as nn
import torch.nn.functional as F

from mmdet.apis import init_backbone_neck

distilled_file_config = 'D:/Senior/lab/mmdetection/projects/BAANet/configs/BAANet_r50_fpn_1x_kaist_new.py'
distilled_checkpoint = 'D:/Senior/lab/mmdetection/projects/BAANet/checkpoints/best_coco_all_iter_3500.pth'

distilled_backbone , distilled_neck = init_backbone_neck(distilled_file_config,distilled_checkpoint)


def FMSE_loss(current_bacth, true_batch):
    batch_size,C,H,W = current_bacth.shape
    device = 'cuda:0'

    # layer_norm = nn.LayerNorm(normalized_shape=(C, H, W)).to(device)
    
    # # mean = torch.zeros(C).to(device)
    # # std = torch.ones(C).to(device)

    # current_bacth = layer_norm(current_bacth)
    # true_batch = layer_norm(true_batch)
    d = torch.sum((current_bacth-true_batch)**2, dim=1, keepdim=True)

    # print(d)
    soft_d = F.softmax(d.view(batch_size, 1, -1), dim=2)
    soft_d = soft_d.view(d.size())

    loss_matrix = soft_d * d

    loss = torch.sum(loss_matrix)
    loss = loss/C

    return loss
    
device = 'cuda:0'

batch_inputs = torch.rand(4,6,512,640).to(device)

x_truth = distilled_backbone(batch_inputs)
x_truth = distilled_neck(x_truth)


for x in x_truth:
    print(x.shape)

x_1 = x_truth[0]

x_2 = x_1 - 0.1

# input_tensor1 = torch.tensor([
#     [
#         [
#             [1, 2, 3],
#             [4, 5, 6],
#             [7, 8, 9]
#         ],
#         [
#             [10, 11, 12],
#             [13, 14, 15],
#             [16, 17, 18]
#         ],
#         [
#             [19, 20, 21],
#             [22, 23, 24],
#             [25, 26, 27]
#         ],
#         [
#             [28, 29, 30],
#             [31, 32, 33],
#             [34, 35, 36]
#         ]
#     ]
# ])
# input_tensor1 = input_tensor1.to(torch.float32)

# input_tensor2 = torch.tensor([
#     [
#         [
#             [1, 2, 3],
#             [4, 5, 6],
#             [7, 8, 9]
#         ],
#         [
#             [10, 11, 12],
#             [13, 14, 15],
#             [16, 17, 18]
#         ],
#         [
#             [19, 20, 21],
#             [22, 23, 24],
#             [25, 26, 27]
#         ],
#         [
#             [28, 29, 30],
#             [31, 32, 33],
#             [32, 33, 34]
#         ]
#     ]
# ])
# input_tensor2 = input_tensor2.to(torch.float32)

distilled_loss = FMSE_loss(x_1,x_2)
print(distilled_loss)