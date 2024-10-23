'''
Author: wds-wsl_ubuntu22 wdsnpshy@163.com
Date: 2024-10-16 14:45:34
LastEditors: wds-wsl_ubuntu22 wdsnpshy@163.com
LastEditTime: 2024-10-23 16:39:21
FilePath: /Data_engineering/test.py
Description: 
微信: 15310638214 
邮箱：wdsnpshy@163.com 
Copyright (c) 2024 by ${wds-wsl_ubuntu22}, All Rights Reserved. 
'''
import os

import cv2
from getPicture import split_and_save_squares
from use_model import MNISTPredictor
from PIL import Image

def main():

    # 放到对应的数字的文件夹下
    predictor = MNISTPredictor(model_path='./best_model.pth')  # 加载训练好的模型
    # 遍历文件夹下的所有图片
    folder = "/home/ubuntu22/workspace/Data_engineering/tools/output_result/"      # 图片所在文件夹
    output_folder = "./tools/output_result" 
    img_names = os.listdir(folder)
    for img in img_names:
        if img.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder, img)
            # 读取图像数据
            image = Image.open(img_path)
            result = predictor.predict(image)
            print(f"图片{img}的预测结果为：{result}")
            
            # 将图片复制保存到对应的文件夹下
            save_folder = os.path.join(output_folder, str(result))
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            save_path = os.path.join(save_folder, img)  # 保存路径
            image.save(save_path)   # 保存图片
            # 显示图片
            img = cv2.imread(img_path)
            cv2.imshow('image', img)
            if cv2.waitKey(0) & 0xFF == ord('q'):
                break
    cv2.destroyAllWindows()



if __name__ == '__main__':
    main()

    # 399620678