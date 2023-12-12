import sys
sys.path.append('/home/yons/czz/mmdetection/projects/BAANet/baanet/utils/complex_light_env_gen/Automold')
import Automold as am
import Helpers as hp
import os
import cv2
import random
from tqdm import tqdm
class ImageProcessor:
    def __init__(self, output_folder):
        self.output_folder = output_folder

    # 定义函数，接受函数 fun 和概率 p 作为参数
    def process_with_p(self, func, p, img):
        # 生成一个随机数，介于 0 和 1 之间
        random_number = random.random()
        
        # 如果随机数小于概率 p，则执行函数 fun
        if random_number < p:
            return func(img)
        else:
            return img

    def aug_img_process(self, img):
        img = self.process_with_p(am.add_sun_flare, 0.8, img)
        img = self.process_with_p(am.brighten, 0.8, img)
        return img
    def process_image(self, input_image_path):
        # 读取输入图像
        img = cv2.imread(input_image_path)
        aug_img= self.aug_img_process(img) ##2 random aug_types are applied in both images
        # 构建输出文件路径
        output_file_path = os.path.join(self.output_folder, os.path.basename(input_image_path))

        # 保存处理后的图像到输出文件夹
        cv2.imwrite(output_file_path, aug_img)

    def process_images(self, input_folder):
        # 遍历输入文件夹中的所有文件
        for root, _, files in os.walk(input_folder):
            for file in tqdm(list(files)):
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    input_image_path = os.path.join(root, file)
                    self.process_image(input_image_path)

if __name__ == "__main__":
    input_folder = "/media/yons/1/yxx/grad_proj_data/KAIST/kaist_test/kaist_test_visible"  # 输入文件夹的路径
    output_folder = "/media/yons/1/yxx/grad_proj_data/KAIST/kaist_test/kaist_test_visible_complex_light_new"  # 输出文件夹的路径

    processor = ImageProcessor(output_folder)
    processor.process_images(input_folder)
