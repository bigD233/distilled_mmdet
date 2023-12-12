import json

# 读取COCO JSON文件
with open('/home/yons/yxx/grad_project/Multimodal-Object-Detection-via-Probabilistic-Ensembling/evalKAIST/KAIST_annotation.json', 'r') as json_file:
    coco_data = json.load(json_file)

# 添加前缀的函数
def add_prefix_to_filenames(coco_data, prefix):
    for image_info in coco_data['images']:
        image_info['im_name'] = prefix + image_info['im_name']

# 添加前缀
prefix_to_add = 'complex_light_new/'  # 你的前缀
add_prefix_to_filenames(coco_data, prefix_to_add)

# 保存修改后的COCO JSON文件
output_path = '/home/yons/yxx/grad_project/Multimodal-Object-Detection-via-Probabilistic-Ensembling/evalKAIST/glare_KAIST_annotation.json'
with open(output_path, 'w') as output_file:
    json.dump(coco_data, output_file)
