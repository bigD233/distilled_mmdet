import os
import json

class TxtToCocoConverter:
    def __init__(self, image_list_file, annotation_folder):
        self.image_list_file = image_list_file
        self.annotation_folder = annotation_folder
        self.coco_data = {
            "images": [],
            "annotations": [],
            "categories": []
        }
        self.category_id_map = {}
        self.annotation_id = 1
        self.trainset_base_pth = "/media/yons/1/yxx/grad_proj_data/KAIST/kaist_train"

    def add_category(self, category_name):
        category_id = len(self.coco_data["categories"]) + 1
        category_info = {
            "id": category_id,
            "name": category_name,
            "supercategory": category_name
        }
        self.coco_data["categories"].append(category_info)
        self.category_id_map[category_name] = category_id

    def convert(self):
        # Read image list file
        with open(self.image_list_file, "r") as image_list:
            image_paths = image_list.read().splitlines()

        for image_path in image_paths:
            set_name, v_name, i_name = image_path.split("/")
            image_filename =  os.path.join(self.trainset_base_pth, set_name, v_name, 'lwir', i_name + '.png')
            image_id = len(self.coco_data["images"]) + 1
            image_info = {
                "id": image_id,
                "file_name": image_filename,
                "width": 640,  # Set the correct width of the image
                "height": 512,  # Set the correct height of the image
                "license": 0,
                "date_captured": "",
            }
            self.coco_data["images"].append(image_info)

            annotation_file = os.path.join(self.annotation_folder, set_name, v_name, 'lwir', i_name+'.txt')
            if os.path.exists(annotation_file):
                with open(annotation_file, "r") as annotation:
                    lines = annotation.read().splitlines()
                    for line in lines[1:]:  # Skip the first line
                        parts = line.split(" ")
                        if len(parts) >= 5:  # Check if it has at least 4 valid values
                            x, y, width, height = map(float, parts[1:5])
                            category_name = "person"
                            category_id = self.category_id_map.get(category_name, None)
                            if category_id is None:
                                self.add_category(category_name)
                                category_id = self.category_id_map[category_name]
                            annotation_info = {
                                "id": self.annotation_id,
                                "image_id": image_id,
                                "category_id": category_id,
                                "bbox": [x, y, width, height],
                                "area": width * height,
                                "segmentation": [],
                                "iscrowd": 0
                            }
                            self.coco_data["annotations"].append(annotation_info)
                            self.annotation_id += 1

    def save_to_json(self, output_file):
        with open(output_file, "w") as json_file:
            json.dump(self.coco_data, json_file)

if __name__ == "__main__":
    image_list_file = "/media/yons/1/yxx/grad_proj_data/KAIST/anno/train_anno/ARCNN_train.txt"
    annotation_folder = "/media/yons/1/yxx/grad_proj_data/KAIST/anno/train_anno/kaist-paired/annotations"

    converter = TxtToCocoConverter(image_list_file, annotation_folder)
    converter.convert()
    converter.save_to_json("/media/yons/1/yxx/grad_proj_data/KAIST/anno/train_anno/ARCNN_KAIST_train_IR_annotation.json")
