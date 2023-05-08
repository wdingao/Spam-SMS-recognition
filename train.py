import torch
import torch.nn as nn
import torch.optim as optim
import data_processor as dpr
import LSTM_Model as mymodel


# 数据集路径
data_set_path = 'data/data_set.txt'

# 模型参数设置
# 输入张量的维度，对于使用BERT提取的词向量来说，通常为768
input_dim = 768
# LSTM隐藏状态的维度
hidden_dim = 128
# output_dim：输出层维度
output_dim = 1
# LSTM的层数
num_layers = 2
# 学习率
learning_rate = 0.01
# 训练轮次
num_epochs = 100

# 模型和优化器初始化
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_dataset, val_dataset, test_dataset, train_data_loader, val_data_loader, test_data_loader = dpr.data_process(data_set_path)
model = mymodel.LSTM(input_dim, hidden_dim, output_dim, num_layers=num_layers).to(device)
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 定义损失函数
criterion = nn.BCELoss()

print("开始训练")

# 训练循环
for epoch in range(num_epochs):
    # 训练阶段
    train_loss = 0.0
    model.train()
    for batch_idx, (data, target) in enumerate(train_data_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target.float())
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_data_loader)

    # 验证阶段
    val_loss = 0.0
    val_acc = 0.0
    model.eval()
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(val_data_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target.float())
            val_loss += loss.item()
            pred = (output >= 0.5).float()
            val_acc += (pred == target).float().sum()
        val_loss /= len(val_data_loader)
        val_acc /= len(val_dataset)

    # 打印训练信息
    print(f"Epoch {epoch + 1}/{num_epochs} Train Loss: {train_loss:.4f} Val Loss: {val_loss:.4f} Val Acc: {val_acc:.4f}")

# 保存整个模型
torch.save(model, 'data/model.pt')
