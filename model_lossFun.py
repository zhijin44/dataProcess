import pandas as pd
import numpy as np
import random
import math
import itertools
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler

# 对于Python的random模块
random.seed(0)
# 对于numpy
np.random.seed(0)
# 对于PyTorch
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)


# 定义优化器和损失函数
class WeightedMSELoss(nn.Module):
    def __init__(self, weight):
        super(WeightedMSELoss, self).__init__()
        self.weight = weight

    def forward(self, input, target):
        # 首先，我们需要确保输入和目标的形状相同
        assert input.shape == target.shape
        # 然后，我们可以计算权重误差
        weighted_squared_error = self.weight * (input[:, 0] - target[:, 0]) ** 2 + (input[:, 1] - target[:, 1]) ** 2
        return weighted_squared_error.mean()


# 2. 模型定义
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(3, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.fc4 = nn.Linear(256, 256)
        self.bn4 = nn.BatchNorm1d(256)
        self.fc5 = nn.Linear(256, 128)
        self.bn5 = nn.BatchNorm1d(128)
        self.fc6 = nn.Linear(128, 64)
        self.bn6 = nn.BatchNorm1d(64)
        self.fc7 = nn.Linear(64, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        x = self.bn3(x)
        x = torch.relu(x)
        x = self.fc4(x)
        x = self.bn4(x)
        x = torch.relu(x)
        x = self.fc5(x)
        x = self.bn5(x)
        x = torch.relu(x)
        x = self.fc6(x)
        x = self.bn6(x)
        x = torch.relu(x)
        x = self.fc7(x)
        return x


# 1. 数据预处理
# 读取csv文件
df = pd.read_csv('unique.csv')
# 将DataFrame转化为numpy数组
data = df.to_numpy()
# 对数据进行分类
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


# 训练循环
model = Net()

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)  # Adam在训练过程中为每个参数保持单独的学习率
# optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-3)  # 1e-5是L2正则化系数
criterion = nn.MSELoss()
# criterion = WeightedMSELoss(weight=100)  # 让con的权重是per的2倍

# 定义L1正则化的系数
lambda_l1 = 0.0001

epoch_num = 400
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)  # 每50个epoch，学习率减半
# scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.5)
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
        # 添加L1正则化
        l1_reg = torch.tensor(0.).to(inputs.device)
        for param in model.parameters():
            l1_reg += torch.norm(param, 1)
        loss += lambda_l1 * l1_reg
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

    # 在每个 epoch 结束后更新学习率
    scheduler.step()

    # 切换到验证模式
    model.eval()
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
