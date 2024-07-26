import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 读取数据
df = pd.read_csv('unique_consecutive.csv')

# 将数据分割成输入和输出
input_data = df[['wifi', '5g', 'any']].values
output_data = df[['con', 'per']].values

# 划分训练集和验证集
input_train, input_valid, output_train, output_valid = train_test_split(input_data, output_data, test_size=0.2)

# 初始化两个StandardScaler对象
input_scaler = StandardScaler()
output_scaler = StandardScaler()
# 使用训练数据对两个scaler进行拟合
input_scaler.fit(input_train)
output_scaler.fit(output_train)
# 使用拟合好的scaler对训练数据和验证数据进行标准化
input_train = input_scaler.transform(input_train)
input_valid = input_scaler.transform(input_valid)
output_train = output_scaler.transform(output_train)
output_valid = output_scaler.transform(output_valid)


class TransformerModel(nn.Module):
    def __init__(self, input_dim, output_dim, dim_feedforward, nhead, num_layers):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        # Encoder
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.input_dim, nhead=nhead,
                                                        dim_feedforward=dim_feedforward)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

        # Decoder
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=self.input_dim, nhead=nhead,
                                                        dim_feedforward=dim_feedforward)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers)

        # Output Layer
        self.out = nn.Linear(self.input_dim, output_dim)

    def forward(self, src):
        src = src.view(-1, 1, self.input_dim)

        memory = self.transformer_encoder(src)
        output = self.transformer_decoder(src, memory)  # using src as tgt for decoder
        output = self.out(output)

        return output.view(-1, self.output_dim)


def train(model, input_data, output_data, optimizer, loss_fn):
    model.train()

    # 将numpy数组转为torch张量
    input_data = torch.from_numpy(input_data).float()
    output_data = torch.from_numpy(output_data).float()

    # 前向传播
    output_pred = model(input_data)
    # print(output_pred)
    # print(output_data)

    # 计算损失
    loss = loss_fn(output_pred, output_data)

    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def validate(model, input_data, output_data, loss_fn):
    model.eval()  # 将模型切换到评估模式

    with torch.no_grad():  # 禁用梯度计算
        # 将numpy数组转为torch张量
        input_data = torch.from_numpy(input_data).float()
        output_data = torch.from_numpy(output_data).float()

        # 前向传播
        output_pred = model(input_data)

        # 计算损失
        loss = loss_fn(output_pred, output_data)

    return loss.item()


def accuracy(output_pred, output_data, output_scaler):
    accuracy_range = 0.03
    total = 0
    with torch.no_grad():
        assert output_pred.size() == output_data.size()
        total += output_pred.size(0)
        print(total)

        # 将预测数据和标签反向标准化
        output_pred = output_scaler.inverse_transform(output_pred.numpy())
        output_data = output_scaler.inverse_transform(output_data.numpy())

        # 将反向标准化后的numpy数组转为torch张量
        output_pred = torch.from_numpy(output_pred)
        output_data = torch.from_numpy(output_data)

        diff = torch.abs((output_pred - output_data) / output_data)
        return torch.sum((diff < accuracy_range).float()).item() / total # 计算准确率


# 定义模型参数
INPUT_DIM = 3  # 输入维度，对应[wifi, 5g, any]
OUTPUT_DIM = 2  # 输出维度，对应[con, per]
NHEAD = 3  # 注意力头的数量
DIM_FEEDFORWARD = 2048  # 前馈神经网络的维度
NUM_LAYERS = 6  # 编码器和解码器的层数

# 初始化模型
model = TransformerModel(INPUT_DIM, OUTPUT_DIM, DIM_FEEDFORWARD, NHEAD, NUM_LAYERS)

# 定义优化器和损失函数
optimizer = optim.Adam(model.parameters(), lr=0.005)
loss_fn = nn.MSELoss()  # 你可以根据任务需求选择适当的损失函数

# 训练模型
EPOCHS = 100  # 迭代次数
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

for epoch in range(EPOCHS):
    train_loss = train(model, input_train, output_train, optimizer, loss_fn)
    valid_loss = validate(model, input_valid, output_valid, loss_fn)

    # 计算训练集和验证集的准确率
    train_accuracy = accuracy(model(torch.from_numpy(input_train).float()), torch.from_numpy(output_train).float(),
                              output_scaler)
    valid_accuracy = accuracy(model(torch.from_numpy(input_valid).float()), torch.from_numpy(output_valid).float(),
                              output_scaler)

    # 更新学习率
    scheduler.step()

    print(
            f"Epoch {epoch}/{EPOCHS}: train loss = {train_loss}, valid loss = {valid_loss}, train accuracy = {train_accuracy}, valid accuracy = {valid_accuracy}")
