'''
Author: wds-wsl_ubuntu22 wdsnpshy@163.com
Date: 2024-10-23 15:10:48
LastEditors: wds-wsl_ubuntu22 wdsnpshy@163.com
LastEditTime: 2024-10-23 17:55:52
FilePath: /Data_engineering/use_model.py
Description: 
微信: 15310638214 
邮箱：wdsnpshy@163.com 
Copyright (c) 2024 by ${wds-wsl_ubuntu22}, All Rights Reserved. 
'''
import os
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from tools.model_wds import LeNet  # 导入自定义的LeNet模型
from tools.model_wds import MobileNetV2  # 导入自定义的MobileNetV2模型

class MNISTPredictor: 
    def __init__(self, model_path='best_model.pth'):
        # 使用GPU或CPU
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("使用设备：", self.device)

        # 定义模型并加载训练好的模型参数
        self.model = MobileNetV2().to(self.device)
        # self.model = LeNet().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        self.model.eval()  # 设置模型为评估模式

        # 定义数据预处理步骤
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))  # 和训练时的归一化保持一致
        ])

    def predict(self, image):
        """
        对输入的图像数据进行预测。

        :param image: 输入的图像数据，类型为PIL.Image.Image或numpy.ndarray
        :return: 预测的数字
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        # 转换为灰度图像
        image = image.convert('L')
        
        # # 反转阈值和颜色，与MNIST数据集颜色一致
        image = 255 - np.array(image)
        image = Image.fromarray(image)

        # 调整大小为28x28像素
        image = image.resize((28, 28))
        # 显示图片
        

        # 数据预处理
        image = self.transform(image).unsqueeze(0)  # 添加批次维度

        # 将图片传递给模型进行预测
        image = image.to(self.device)
        outputs = self.model(image)
        _, predicted = torch.max(outputs.data, 1)
        # 如果置信度大于0.5，返回预测结果，否则返回None
        if torch.max(torch.nn.functional.softmax(outputs.data, dim=1)) > 0.7:
            # 返回结果以及图片
            return predicted.item()
        else:
            return None
        # return predicted.item()

if __name__ == "__main__":
    predictor = MNISTPredictor(model_path='./best_model.pth')  # 加载训练好的模型
    # 遍历文件夹下的所有图片
    folder = "./tools/output"
    img_names = os.listdir(folder)
    for img in img_names:
        if img.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder, img)   
            # 读取图像数据
            image = Image.open(img_path)
            result = predictor.predict(image)
            print(f"图片{img}的预测结果为：{result}")
            # 显示图片
            img = cv2.imread(img_path)
            cv2.imshow('image', img)
            if cv2.waitKey(0) & 0xFF == ord('q'):
                break
    cv2.destroyAllWindows()