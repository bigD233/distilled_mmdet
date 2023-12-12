import json
import shutil
import os

def expand_and_append(coco_json_path, condition_func, output_json_path, num_copies):
    with open(coco_json_path, 'r') as file:
        coco_data = json.load(file)

    new_images = []
    new_annotations = []

    for image in coco_data['images']:
        if condition_func(image['file_name']):
            selected_annotations = [anno for anno in coco_data['annotations'] if anno['image_id'] == image['id']]

            # 复制图像和注释多份
            for i in range(num_copies):
                new_image = image.copy()
                # new_image['id'] = len(coco_data['images']) + len(new_images) + 1
                new_image['id'] = len(new_images) + 1
                new_images.append(new_image)

                for annotation in selected_annotations:
                    new_annotation = annotation.copy()
                    # new_annotation['id'] = len(coco_data['annotations']) + len(new_annotations) + 1
                    new_annotation['id'] = len(new_annotations) + 1
                    new_annotation['image_id'] = new_image['id']
                    new_annotations.append(new_annotation)

    # 将新图像和注释添加到原始数据中
    # coco_data['images'].extend(new_images)
    # coco_data['annotations'].extend(new_annotations)

    coco_data['images'] = new_images
    coco_data['annotations'] = new_annotations

    # 写入新的 JSON 文件
    with open(output_json_path, 'w') as output_file:
        json.dump(coco_data, output_file, indent=4)

# 示例用法
coco_json_path = '/media/yons/1/yxx/grad_proj_data/KAIST/anno/train_anno/KAIST_train_RGB_annotation.json'
output_json_path = '/media/yons/1/yxx/grad_proj_data/KAIST/anno/train_anno/KAIST_train_RGB_annotation_only_day.json'

# 自定义条件函数，例如：图像路径包含特定字符串
def condition_func(file_name):
    return any(substring in file_name for substring in ['set00', 'set01', 'set02'])

num_copies = 1  # 替换为你想要拓展的份数

expand_and_append(coco_json_path, condition_func, output_json_path, num_copies)
