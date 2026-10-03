import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from convlstm import ConvLSTM

# 加载数据
X = np.load("D:/Weather_Project/data/X.npy")
y = np.load("D:/Weather_Project/data/y.npy")

assert X.shape[1:] == (7, 6) and y.shape[1:] == (6,), "数据形状不对，请重新运行 preprocess.py"

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

model = TempConvLSTM()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 加权损失：气温权重3，最高最低2，其他1
weights = torch.FloatTensor([3.0, 2.0, 2.0, 1.0, 1.0, 1.0])
criterion = nn.MSELoss(reduction='none')

best_test_mse = float('inf')

print("开始训练...")
for epoch in range(100):
    model.train()
    total_loss = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        pred = model(batch_X)
        loss = (criterion(pred, batch_y) * weights).mean()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if (epoch + 1) % 5 == 0:
        model.eval()
        with torch.no_grad():
            test_pred = model(X_test_t)
            # 只用气温（第0列）计算MSE作为早停指标
            test_mse_temp = nn.MSELoss()(test_pred[:, 0], y_test_t[:, 0]).item()
            # 加权总MSE作为打印指标
            test_loss = (criterion(test_pred, y_test_t) * weights).mean().item()
        print(f"Epoch {epoch+1:3d} | 加权Loss: {total_loss/len(train_loader):.4f} | 气温MSE: {test_mse_temp:.4f}")
        
        if test_mse_temp < best_test_mse:
            best_test_mse = test_mse_temp
            torch.save(model.state_dict(), "D:/Weather_Project/output/best_convlstm_model.pth")
            print(f"🌟 新最佳气温MSE={test_mse_temp:.4f}，已保存")

print(f"训练完成！最佳气温MSE={best_test_mse:.4f}")