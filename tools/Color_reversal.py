'''
Author: wds-wsl_ubuntu22 wdsnpshy@163.com
Date: 2024-10-23 15:48:30
LastEditors: wds-wsl_ubuntu22 wdsnpshy@163.com
LastEditTime: 2024-10-23 15:53:25
FilePath: /Data_engineering/Split_digit/tools/Color_reversal.py
Description: 反转阈值和颜色，与minist数据集颜色一致
微信: 15310638214 
邮箱：wdsnpshy@163.com 
Copyright (c) 2024 by ${wds-wsl_ubuntu22}, All Rights Reserved. 
'''
# 遍历文件夹下的图片，将图片颜色反转
import os
import cv2
import numpy as np
import tqdm

def color_reversal(input_dir, output_dir):
    img_name = os.listdir(input_dir)
    total = len(img_name)
    print(f"共有{total}张图片需要处理")


    for img in tqdm.tqdm(img_name):
        img_path = os.path.join(input_dir, img)
        image = cv2.imread(img_path)
        # 颜色反转
        image = 255 - image
        # 保存图片
        cv2.imwrite(os.path.join(output_dir, img), image)
        
if __name__ == "__main__":
    input_dir = 'output/'
    output_dir = 'output_reversal/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    color_reversal(input_dir, output_dir)
    print("图片颜色反转完成！")