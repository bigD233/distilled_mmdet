import json
import copy

def duplicate_and_modify_coco_json(input_json_path, output_json_path):
    with open(input_json_path, 'r') as f:
        coco_data = json.load(f)

    # 复制img和anno
    duplicated_images = copy.deepcopy(coco_data['images'])
    duplicated_annotations = copy.deepcopy(coco_data['annotations'])

    # 额外复制的img的info，将path字符串之前加上"complex_light_new"，并更改ID
    new_img_id_offset = max(img['id'] for img in coco_data['images']) + 1
    for img_info in duplicated_images:
        img_info['im_name'] = "complex_light_new/" + img_info['im_name']
        img_info['id'] += new_img_id_offset

    # 额外复制的anno，并更改ID
    new_anno_id_offset = max(anno['id'] for anno in coco_data['annotations']) + 1
    for anno_info in duplicated_annotations:
        anno_info['id'] += new_anno_id_offset
        anno_info['image_id'] += new_img_id_offset

    # 添加在原文件的后面
    coco_data['images'].extend(duplicated_images)
    coco_data['annotations'].extend(duplicated_annotations)

    # 写入新的JSON文件
    with open(output_json_path, 'w') as f:
        json.dump(coco_data, f, indent=4)

if __name__ == "__main__":
    input_json_path = "/home/yons/yxx/grad_project/Multimodal-Object-Detection-via-Probabilistic-Ensembling/evalKAIST/KAIST_annotation.json"
    output_json_path = "/home/yons/yxx/grad_project/Multimodal-Object-Detection-via-Probabilistic-Ensembling/evalKAIST/KAIST_annotation_with_complex_light.json"

    duplicate_and_modify_coco_json(input_json_path, output_json_path)