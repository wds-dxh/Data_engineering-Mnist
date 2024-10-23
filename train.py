'''
Author: wds-wsl_ubuntu22 wdsnpshy@163.com
Date: 2024-10-16 14:45:34
LastEditors: wds-wsl_ubuntu22 wdsnpshy@163.com
LastEditTime: 2024-10-23 17:43:08
FilePath: /Data_engineering/train.py
Description: 智能网联汽车数据工程-训练模型----lenet和mobilenetv2
微信: 15310638214 
邮箱：wdsnpshy@163.com 
Copyright (c) 2024 by ${wds-wsl_ubuntu22}, All Rights Reserved. 
'''
import torch
import torch.nn as nn       # 神经网络模块
import torch.optim as optim # 优化器模块
import torchvision          # 计算机视觉模块
import torchvision.transforms as transforms # 数据预处理模块
from tools.model_wds import LeNet # 导入自定义的LeNet模型
from tools.model_wds import MobileNetV2 # 导入自定义的MobileNetV2模型
from tqdm import tqdm

# 如果有GPU就用GPU，没有就用CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("使用设备：", device)

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(), # 将图片转换为tensor
    transforms.Normalize((0.5,), (0.5,)) # 归一化
])

# 从本地加载MNIST数据集
training_dataset = torchvision.datasets.MNIST(
    root='./mnist',
    train=True,
    download=False,
    transform=transform
)
test_dataset = torchvision.datasets.MNIST(
    root='./mnist',
    train=False,
    download=False,
    transform=transform
)

# 数据加载器
training_loader = torch.utils.data.DataLoader(
    dataset=training_dataset,
    batch_size=64,
    shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset,
    batch_size=64,
    shuffle=False
)

# 定义网络结构，直接使用PyTorch提供的模型 == LeNet: 两个卷积层，两个池化层，三个全连接层
model = LeNet().to(device)

# 使用更好的MobileNetV2模型,修改输出层为10
# model = MobileNetV2().to(device)
# 加载本地模型


print(model)
# 定义损失函数，优化器等
criterion = nn.CrossEntropyLoss()                       # 交叉熵损失函数
optimizer = optim.Adam(model.parameters(), lr=0.001)    # Adam优化器

# 训练模型-使用tqdm显示进度条
EPOCHS = 50
for epoch in tqdm(range(EPOCHS)):
    running_loss = 0.0
    for i, data in enumerate(training_loader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / len(training_loader)))

    # 每个epoch结束后测试一下模型的准确率
    correct = 0 # 预测正确的图片数
    total = 0   # 总共的图片数
    correct_over = 0 # 记录最高的准确率
    with torch.no_grad():       #使用with，不会计算梯度，节省内
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
        # 保存精度最高的模型
        if correct > correct_over:
            correct_over = correct
            torch.save(model.state_dict(), 'best_model_lenet.pth')
            print('Model saved!')

print('Finished Training')
        
        
            