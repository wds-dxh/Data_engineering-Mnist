'''
Author: wds-wsl_ubuntu22 wdsnpshy@163.com
Date: 2024-10-16 14:47:33
LastEditors: wds-wsl_ubuntu22 wdsnpshy@163.com
LastEditTime: 2024-10-16 14:58:55
FilePath: /Data_engineering/download_minist.py
Description: 如果没有下载过mnist数据集，可以运行这个脚本下载
微信: 15310638214 
邮箱：wdsnpshy@163.com 
Copyright (c) 2024 by ${wds-wsl_ubuntu22}, All Rights Reserved. 
'''

import sys, os
sys.path.insert(0, os.getcwd())

from torchvision.datasets import MNIST
import PIL
from tqdm import tqdm       # 显示进度条

if __name__ == "__main__":
    # 图片保存路径
    root = 'mnist_jpg'
    if not os.path.exists(root):
        os.makedirs(root)

    # 从网络上下载或从本地加载MNIST数据集
    # 训练集60K、测试集10K
    # torchvision.datasets.MNIST接口下载的数据一组元组
    # 每个元组的结构是: (PIL.Image.Image image model=L size=28x28, 标签数字 int)
    # 如果没有下载过mnist数据集，可以运行这个脚本下载, 如果有下载过，可以注释掉这个脚本
    # training_dataset = MNIST(
    #     root='mnist',
    #     train=True,
    #     download=True,
    # )
    # test_dataset = MNIST(
    #     root='mnist',
    #     train=False,
    #     download=True,
    # )

    # 加载本地数据集
    training_dataset = MNIST(
        root='mnist',
        train=True,
        download=False,
    )
    test_dataset = MNIST(
        root='mnist',
        train=False,
        download=False,
    )

    for idx, (X, y) in enumerate(training_dataset):
        print(X, y)
        break

    # # 保存训练集图片
    # with tqdm(total=len(training_dataset), ncols=150) as pro_bar:
    #     for idx, (X, y) in enumerate(training_dataset):
    #         f = root + "/" + "training_" + str(idx) + \
    #             "_" + str(training_dataset[idx][1] )+ ".jpg"  # 文件路径
    #         training_dataset[idx][0].save(f)
    #         pro_bar.update(n=1)

    # # 保存测试集图片
    # with tqdm(total=len(test_dataset), ncols=150) as pro_bar:
    #     for idx, (X, y) in enumerate(test_dataset):
    #         f = root + "/" + "test_" + str(idx) + \
    #             "_" + str(test_dataset[idx][1] )+ ".jpg"  # 文件路径
    #         test_dataset[idx][0].save(f)
    #         pro_bar.update(n=1)
