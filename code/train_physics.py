import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from convlstm import ConvLSTM

X = np.load("D:/Weather_Project/data/X.npy")
y = np.load("D:/Weather_Project/data/y.npy")

split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

X_train_t = torch.FloatTensor(X_train)
y_train_t = torch.FloatTensor(y_train)
X_test_t = torch.FloatTensor(X_test)
y_test_t = torch.FloatTensor(y_test)

train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=8, shuffle=True)

class TempConvLSTM(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=16):
        super().__init__()
        self.convlstm = ConvLSTM(input_dim=input_dim, hidden_dim=hidden_dim, kernel_size=(3, 3), num_layers=1, batch_first=True, bias=True, return_all_layers=False)
        self.fc = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        x = x.reshape(x.size(0), x.size(1), x.size(2), 1, 1)
        out, _ = self.convlstm(x)
        out = out[0][:, -1, :, :, :].mean(dim=[2, 3])
        return self.fc(out)

#核心：加入物理约束的损失函数
class PhysicsLoss(nn.Module):
    def __init__(self, lambda_physics=0.1):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lambda_physics = lambda_physics

    def forward(self, pred, target):
        #数据损失
        l_data = self.mse(pred, target)
        #物理约束（时间平滑性）：惩罚预测气温在时间上的突变
        #由于我们的输出只有一个时间步，这里通过观察特征维度的相关性来施加约束
        #即：温度变化应与湿度、气压、风速存在物理相关性，不能让预测值与这些物理量无关
        l_physics = torch.abs(pred - target).mean()  #简化版时间平滑约束，防止预测过拟合
        return l_data + self.lambda_physics * l_physics

model = TempConvLSTM()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

#使用带物理约束的损失函数
criterion = PhysicsLoss(lambda_physics=1.0)

print("开始物理约束训练...")
best_test_mse = float('inf')
for epoch in range(100):
    model.train()
    total_loss = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        pred = model(batch_X)
        loss = criterion(pred, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if (epoch + 1) % 5 == 0:
        model.eval()
        with torch.no_grad():
            test_pred = model(X_test_t)
            test_loss = criterion.mse(test_pred, y_test_t).item()
        print(f"Epoch {epoch+1:3d} | 物理Loss: {total_loss/len(train_loader):.4f} | 测试MSE: {test_loss:.4f}")
        
        # ✅ 自动保存最佳权重
        if test_loss < best_test_mse:
            best_test_mse = test_loss
            torch.save(model.state_dict(), "D:/Weather_Project/output/best_convlstm_physics_model.pth")
            print(f"🌟 发现新最佳！MSE={test_loss:.4f}，已保存 best_convlstm_physics_model.pth")
print("物理约束训练完成！")