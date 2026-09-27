import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from convlstm import ConvLSTM

# 1. 加载数据
X = np.load("D:/Weather_Project/data/X.npy")  # 形状 (4011, 7, 6)
y = np.load("D:/Weather_Project/data/y.npy")  # 形状 (4011, 1)

# 2. 划分训练集和测试集
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

X_train_t = torch.FloatTensor(X_train)
y_train_t = torch.FloatTensor(y_train)
X_test_t = torch.FloatTensor(X_test)
y_test_t = torch.FloatTensor(y_test)

# 纯CPU运行，Batch调小一点以防卡死
train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=8, shuffle=True)

# 3. 定义 ConvLSTM 模型
class TempConvLSTM(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=16): # 因为没显卡，hidden_dim用16就够了
        super().__init__()
        self.convlstm = ConvLSTM(
            input_dim=input_dim, hidden_dim=hidden_dim,
            kernel_size=(3, 3), num_layers=1, # num_layers设为1，减少计算量
            batch_first=True, bias=True, return_all_layers=False
        )
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = x.reshape(x.size(0), x.size(1), x.size(2), 1, 1)
        out, _ = self.convlstm(x)
        out = out[0][:, -1, :, :, :].mean(dim=[2, 3])
        return self.fc(out)

model = TempConvLSTM()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# 4. 开始训练
print("开始训练！由于是CPU，可能需要几分钟，请耐心等待...")
for epoch in range(100): # 先跑30轮看看效果
    model.train()
    total_loss = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        pred = model(batch_X)
        loss = criterion(pred, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if (epoch + 1) % 5 == 0: # 每5轮打印一次
        model.eval()
        with torch.no_grad():
            test_pred = model(X_test_t)
            test_loss = criterion(test_pred, y_test_t).item()
        print(f"Epoch {epoch+1:3d} | 训练Loss: {total_loss/len(train_loader):.4f} | 测试MSE: {test_loss:.4f}")

# 5. 保存模型
torch.save(model.state_dict(), "D:/Weather_Project/output/convlstm_model.pth")
print("训练完成！模型已保存至 D:/Weather_Project/output/convlstm_model.pth")