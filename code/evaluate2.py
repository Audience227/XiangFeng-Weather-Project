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
    print("✅ 成功加载无约束模型")
except:
    print("❌ 警告: 未找到无约束模型文件")

try:
    model_physics.load_state_dict(torch.load(model_physics_path))
    model_physics.eval()
    print("✅ 成功加载有约束模型")
except:
    print("❌ 警告: 未找到物理约束模型文件")

# ================== 5. 预测与计算 ==================
with torch.no_grad():
    y_pred_base = model_base(X_test_t).numpy()       # 无约束预测
    y_pred_physics = model_physics(X_test_t).numpy() # 有约束预测

# 普通天气和极端天气的MSE对比
mse_base = np.mean((y_test - y_pred_base)**2)
mse_physics = np.mean((y_test - y_pred_physics)**2)
print(f"全局模型 MSE - 无约束: {mse_base:.4f}, 有约束: {mse_physics:.4f}")

# ⚠️ 定义极端天气：标准化后 > 1.0 代表高温（具体对应多少度取决于你标准化时的均值）
# 如果你想知道具体的阈值，可以修改这个 1.0 值，或者去 preprocess.py 里逆标准化。
extreme_threshold = 1.0 
extreme_idx = np.where(y_test > extreme_threshold)[0]

if len(extreme_idx) > 0:
    mse_extreme_base = np.mean((y_test[extreme_idx] - y_pred_base[extreme_idx])**2)
    mse_extreme_physics = np.mean((y_test[extreme_idx] - y_pred_physics[extreme_idx])**2)
    print(f"极端天气样本数: {len(extreme_idx)} | 无约束 MSE: {mse_extreme_base:.4f} | 有约束 MSE: {mse_extreme_physics:.4f}")
else:
    print("⚠️ 测试集中没有发现极端天气样本。")

# ================== 6. 画图（带极端天气高亮） ==================
n_show = 200  # 展示前200个测试样本
plt.figure(figsize=(15, 6))

# 画三条线
plt.plot(y_test[:n_show], label='真实气温 (Ground Truth)', color='blue', linewidth=2, alpha=0.7)
plt.plot(y_pred_base[:n_show], label='无约束ConvLSTM预测', color='red', linestyle='--', linewidth=1.5)
plt.plot(y_pred_physics[:n_show], label='物理约束ConvLSTM预测', color='green', linestyle='-.', linewidth=1.5)

# 🌟 新增：高亮极端天气时段
# 找出前 n_show 个样本里属于极端天气的索引
extreme_show_idx = [i for i in extreme_idx if i < n_show]
if len(extreme_show_idx) > 0:
    # 为了防止连线过于密集，我们不用折线，用红色的散点或阴影框
    # 这里采用散点标记，并在图例中说明
    plt.scatter(extreme_show_idx, y_test[extreme_show_idx], color='orange', s=40, zorder=5, label='极端高温样本 (>1.0标准差)')
    
    # 可选：在最高温那个点旁边加一句注释
    plt.text(extreme_show_idx[0], y_test[extreme_show_idx[0]] + 0.3, 
             f"极端天气样本\n无约束MSE: {mse_extreme_base:.4f}\n有约束MSE: {mse_extreme_physics:.4f}", 
             fontsize=10, color='darkred', bbox=dict(facecolor='white', alpha=0.7, edgecolor='red'))

plt.title('气温预测模型对比：无约束 vs 物理约束 (含极端天气高亮)', fontsize=15)
plt.xlabel('测试样本序号', fontsize=12)
plt.ylabel('标准化气温', fontsize=12)
plt.legend(fontsize=11, loc='upper right')
plt.grid(True, alpha=0.3)
plt.tight_layout()

# ================== 7. 智能命名与保存 ==================
base_name = os.path.basename(model_base_path).replace('.pth', '')
physics_name = os.path.basename(model_physics_path).replace('.pth', '')
timestamp = datetime.datetime.now().strftime("%m%d_%H%M%S")
save_path = f"D:/Weather_Project/output/极端天气对比_{base_name}_vs_{physics_name}_{timestamp}.png"

plt.savefig(save_path, dpi=150)
print(f"📸 带极端天气高亮的对比图保存成功！路径: {save_path}")

# ========== 记录 evaluate2.py 的本次运行时间 ==========
import os, time
output_dir = "D:/Weather_Project/output"
with open(os.path.join(output_dir, "time_extreme.txt"), "w") as f:
    f.write(str(time.time()))
print("📝 evaluate2.py 运行时间已记录")