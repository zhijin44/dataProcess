import pandas as pd
import numpy as np
import random
import math
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler


# 2. 模型定义 - CNN
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv1d(3, 64, kernel_size=1, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=1, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        self.layer3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=1, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = x.unsqueeze(2)  # reshape from (batch_size, features) to (batch_size, features, 1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(x.size(0), -1)  # flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# class ConvNet(nn.Module):
#     def __init__(self):
#         super(ConvNet, self).__init__()
#
#         self.layer1 = nn.Sequential(
#             nn.Conv1d(3, 64, kernel_size=1, stride=1, padding=1),
#             nn.BatchNorm1d(64),
#             nn.ReLU(),
#             nn.MaxPool1d(kernel_size=2, stride=2)
#         )
#
#         self.layer2 = nn.Sequential(
#             nn.Conv1d(64, 128, kernel_size=1, stride=1, padding=1),
#             nn.BatchNorm1d(128),
#             nn.ReLU(),
#             nn.MaxPool1d(kernel_size=2, stride=2)
#         )
#
#         self.fc = nn.Linear(128, 2)
#
#     def forward(self, x):
#         x = x.unsqueeze(2)  # reshape from (batch_size, features) to (batch_size, features, 1)
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = x.view(x.size(0), -1)  # flatten the tensor
#         x = self.fc(x)
#         return x


# 1. 数据预处理
# 读取csv文件
df = pd.read_csv('unique.csv')
data = df.to_numpy()

groups = itertools.groupby(data, key=lambda x: x[0])
train_data = []
test_data = []
# 对每一类的数据进行划分
for _, group in groups:
    group = list(group)
    random.shuffle(group)  # 在进行划分之前先打乱数据
    train_size = math.ceil(len(group) * 0.8)  # 向上取整  # 计算训练集的大小
    train_data.extend(group[:train_size])
    test_data.extend(group[train_size:])

# 获取输入和输出，提取前两维作为输出（目标），后三维作为输入
train_inputs = np.array(train_data)[:, -3:]
train_targets = np.array(train_data)[:, :2]
# print(train_inputs)
test_inputs = np.array(test_data)[:, -3:]
test_targets = np.array(test_data)[:, :2]
# print(test_inputs)

# 数据标准化 （执行Z-score Normalization，将输入数据标准化，使得每一列特征的平均值为0，标准差为1）
scaler = StandardScaler()
train_inputs = scaler.fit_transform(train_inputs)
test_inputs = scaler.fit_transform(test_inputs)

# 创建PyTorch数据加载器
# print(train_inputs)
train_dataset = TensorDataset(torch.tensor(train_inputs).float(), torch.tensor(train_targets).float())
train_loader = DataLoader(train_dataset, batch_size=64)

val_dataset = TensorDataset(torch.tensor(test_inputs).float(), torch.tensor(test_targets).float())
val_loader = DataLoader(val_dataset, batch_size=64)

# 2. 训练循环
model = ConvNet()

# 定义优化器 / scheduler / loss function
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)  # Adam在训练过程中为每个参数保持单独的学习率
# optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-3)  # 1e-5是L2正则化系数
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)  # 每50个epoch，学习率减半
criterion = nn.MSELoss()

epoch_num = 400
accuracy_range_per = 0.05
for epoch in range(epoch_num):
    model.train()  # 切换到训练模式
    train_loss, correct, total = 0, 0, 0
    for inputs, targets in train_loader:
        # 前向传播
        outputs = model(inputs)
        # print(outputs, targets)

        # 计算训练损失
        loss = criterion(outputs, targets)
        train_loss += loss.item()

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 计算训练集准确率
        total += targets.size(0)
        correct += torch.sum(torch.abs(outputs - targets) / targets < accuracy_range_per).item()

    train_loss /= len(train_loader)
    accuracy_train = correct / total

    model.eval()  # 切换到验证模式
    val_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            outputs = model(inputs)

            # 计算验证损失
            loss = criterion(outputs, targets)
            val_loss += loss.item()

            # 计算验证集准确率
            total += targets.size(0)
            correct += torch.sum(torch.abs(outputs - targets) / targets < accuracy_range_per).item()

        val_loss /= len(val_loader)
        accuracy_val = correct / total

    if epoch % 5 == 0:
        print(
            f"Epoch: {epoch}, Train Loss: {train_loss:.6f}, Val. Loss: {val_loss:.6f}, Train Acc.: {accuracy_train:.6f}, Val. Acc.: {accuracy_val:6f}")
