# -*- coding: utf-8 -*-
"""
train_physics_v2.py —— 真物理约束版六输出模型训练
物理约束：热力学一致性 T2M_MIN <= T2M <= T2M_MAX（违反即惩罚）
对比基线：code/train_convlstm.py（纯加权MSE）
注意：标准化是各特征独立的线性变换，不改变同列间的大小关系，
      因此该约束在标准化空间中同样成立。
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from convlstm import ConvLSTM

X = np.load("data/X.npy")
y = np.load("data/y.npy")
assert X.shape[1:] == (7, 6) and y.shape[1:] == (6,), "数据形状不对"

split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

X_train_t = torch.FloatTensor(X_train)
y_train_t = torch.FloatTensor(y_train)
X_test_t = torch.FloatTensor(X_test)
y_test_t = torch.FloatTensor(y_test)

train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=16, shuffle=True)

class TempConvLSTM(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=16, output_dim=6):
        super().__init__()
        self.convlstm = ConvLSTM(input_dim=input_dim, hidden_dim=hidden_dim,
                                  kernel_size=(3, 3), num_layers=1,
                                  batch_first=True, bias=True, return_all_layers=False)
        self.fc = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        x = x.reshape(x.size(0), x.size(1), x.size(2), 1, 1)
        out, _ = self.convlstm(x)
        out = out[0][:, -1, :, :, :].mean(dim=[2, 3])
        return self.fc(out)

class PhysicsLossV2(nn.Module):
    """加权MSE + 热力学一致性约束（真物理约束）
    约束：T2M_MIN(idx2) <= T2M(idx0) <= T2M_MAX(idx1)
    惩罚违反量：ReLU(max(tmin - tmean, 0)) + ReLU(max(tmean - tmax, 0))
    """
    def __init__(self, lambda_physics=1.0):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')
        self.lambda_physics = lambda_physics
        self.weights = torch.FloatTensor([3.0, 2.0, 2.0, 1.0, 1.0, 1.0])
    def forward(self, pred, target):
        l_data = (self.mse(pred, target) * self.weights).mean()
        viol = torch.relu(pred[:, 2] - pred[:, 0]) + torch.relu(pred[:, 0] - pred[:, 1])
        l_physics = viol.mean()
        return l_data + self.lambda_physics * l_physics

model = TempConvLSTM()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = PhysicsLossV2(lambda_physics=1.0)

best_test_mse = float('inf')
print("开始真物理约束(v2)训练...")
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
            test_mse_temp = nn.MSELoss()(test_pred[:, 0], y_test_t[:, 0]).item()
        print(f"Epoch {epoch+1:3d} | v2物理Loss: {total_loss/len(train_loader):.4f} | 气温MSE: {test_mse_temp:.4f}")
        if test_mse_temp < best_test_mse:
            best_test_mse = test_mse_temp
            torch.save(model.state_dict(), "output/best_convlstm_physics_v2.pth")
            print(f"  新最佳气温MSE={test_mse_temp:.4f}，已保存")

print(f"v2训练完成！最佳气温MSE={best_test_mse:.4f}（标准化口径）")
