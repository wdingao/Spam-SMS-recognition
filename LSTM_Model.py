import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 模型定义
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device)  # 初始化隐层状态
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device)  # 初始化细胞状态
        out, _ = self.lstm(x, (h0, c0))  # lstm输出
        out = self.fc(out[:, -1, :])  # 取最后一个时序的输出
        out = self.sigmoid(out)  # 用sigmoid函数做二分类
        return out
