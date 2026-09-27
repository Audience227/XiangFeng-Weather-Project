import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import datetime
from convlstm import ConvLSTM

# 设置中文字体，防止图中出现方块
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ================== 1. 定义模型路径 ==================
model_base_path = "D:/Weather_Project/output/convlstm_model.pth"
model_physics_path = "D:/Weather_Project/output/convlstm_physics_model.pth"

# ================== 2. 加载数据 ==================
X = np.load("D:/Weather_Project/data/X.npy")
y = np.load("D:/Weather_Project/data/y.npy")

split = int(len(X) * 0.8)
X_test = X[split:]
y_test = y[split:]
X_test_t = torch.FloatTensor(X_test)

# ================== 3. 定义模型结构 ==================
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

# ================== 4. 加载模型 ==================
model_base = TempConvLSTM()
model_physics = TempConvLSTM()

try:
    model_base.load_state_dict(torch.load(model_base_path))
    model_base.eval()
    print("✅ 成功加载纯数据模型:", os.path.basename(model_base_path))
except:
    print("❌ 警告: 未找到纯数据模型文件")

try:
    model_physics.load_state_dict(torch.load(model_physics_path))
    model_physics.eval()
    print("✅ 成功加载物理约束模型:", os.path.basename(model_physics_path))
except:
    print("❌ 警告: 未找到物理约束模型文件")

# ================== 5. 预测与计算 ==================
with torch.no_grad():
    y_pred_base = model_base(X_test_t).numpy()
    y_pred_physics = model_physics(X_test_t).numpy()

mse_base = np.mean((y_test - y_pred_base)**2)
mse_physics = np.mean((y_test - y_pred_physics)**2)
print(f"纯数据模型 测试集MSE: {mse_base:.4f}")
print(f"物理约束模型 测试集MSE: {mse_physics:.4f}")

# ================== 6. 绘图 ==================
n_show = 200
plt.figure(figsize=(14, 5))
plt.plot(y_test[:n_show], label='真实气温 (Ground Truth)', color='blue', linewidth=2, alpha=0.7)
plt.plot(y_pred_base[:n_show], label='纯ConvLSTM预测', color='red', linestyle='--', linewidth=1.5)
plt.plot(y_pred_physics[:n_show], label='物理约束ConvLSTM预测', color='green', linestyle='-.', linewidth=1.5)

plt.title('气温预测模型对比 (测试集前200个样本)', fontsize=14)
plt.xlabel('测试样本序号', fontsize=12)
plt.ylabel('标准化气温', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# ================== 7. 智能命名与保存（核心改动） ==================
# 自动从路径中提取模型名字（去掉后缀.pth）
base_name = os.path.basename(model_base_path).replace('.pth', '')
physics_name = os.path.basename(model_physics_path).replace('.pth', '')

# 生成当前时间戳（精确到秒，防止覆盖）
timestamp = datetime.datetime.now().strftime("%m%d_%H%M%S")

# 拼接最终文件名
save_path = f"D:/Weather_Project/output/对比图_{base_name}_vs_{physics_name}_{timestamp}.png"

plt.savefig(save_path, dpi=150)
print(f"📸 智能保存成功！图片路径为: {save_path}")