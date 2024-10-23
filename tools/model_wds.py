'''
Author: wds-wsl_ubuntu22 wdsnpshy@163.com
Date: 2024-10-16 15:13:39
LastEditors: wds-wsl_ubuntu22 wdsnpshy@163.com
LastEditTime: 2024-10-23 15:29:36
FilePath: /Data_engineering/model_wds.py
Description: 定义lenet以及更高精度的mobileNet
微信: 15310638214 
邮箱：wdsnpshy@163.com 
Copyright (c) 2024 by ${wds-wsl_ubuntu22}, All Rights Reserved. 
'''
import torch
import torch.nn as nn       # 神经网络模块
import torchvision.models as models # 计算机视觉模块

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # 定义网络结构
        self.conv1 = nn.Conv2d(1, 6, 5) # 输入通道数，输出通道数，卷积核大小
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*4*4, 120) # 输入通道数，输出通道数
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    
    def forward(self, x):
        # 定义前向传播
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(-1, 16*4*4)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# 使用更好的MobileNetV2模型,修改输出层为10
# 使用更好的MobileNetV2模型，修改输出层为10，并调整输入通道为1
class MobileNetV2(nn.Module):
    def __init__(self):
        super(MobileNetV2, self).__init__()
        # 定义MobileNetV2模型
        self.model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

        # 修改第一层卷积层，输入通道改为1（灰度图像），保持原来的输出通道数和卷积核大小
        self.model.features[0][0] = nn.Conv2d(
            in_channels=1,        # 输入通道数从3改为1
            out_channels=self.model.features[0][0].out_channels,
            kernel_size=self.model.features[0][0].kernel_size,
            stride=self.model.features[0][0].stride,
            padding=self.model.features[0][0].padding,
            bias=False
        )

        # 修改分类器的最后一层，将输出改为10类
        self.model.classifier[1] = nn.Linear(1280, 10)
    
    def forward(self, x):
        return self.model(x)
