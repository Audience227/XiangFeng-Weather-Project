import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
from convlstm import ConvLSTM

# ===== 中文字体（云端Linux无黑体，会自动回退到英文，不影响） =====
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

st.set_page_config(page_title="相风 - 南京微气候预测", layout="wide")
st.title("🌬️ 相风 - 南京微气候气温预测平台")

st.markdown("""
> **项目介绍**：本项目基于ConvLSTM及物理约束，对南京地区历史气象数据（2015-2025）进行学习，实现未来短时气温预测与可视化。
""")

# ==================== 模型定义 ====================
class TempConvLSTM(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=16):
        super().__init__()
        self.convlstm = ConvLSTM(input_dim=input_dim, hidden_dim=hidden_dim,
                                  kernel_size=(3, 3), num_layers=1,
                                  batch_first=True, bias=True, return_all_layers=False)
        self.fc = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        x = x.reshape(x.size(0), x.size(1), x.size(2), 1, 1)
        out, _ = self.convlstm(x)
        out = out[0][:, -1, :, :, :].mean(dim=[2, 3])
        return self.fc(out)

# ==================== 加载模型 ====================
@st.cache_resource
def load_model():
    model = TempConvLSTM()
    model.load_state_dict(torch.load("output/convlstm_physics_model.pth", map_location='cpu'))
    model.eval()
    return model

model = load_model()

# ==================== 第一部分：数据预览 ====================
st.subheader("1️⃣ 数据预览")
try:
    df_demo = pd.read_csv("data/nanjing.csv", skiprows=17)
    df_show = df_demo.head(10).copy()
    df_show.columns = ['年份', '年积日', '平均气温(℃)', '最高气温(℃)', '最低气温(℃)',
                       '相对湿度(%)', '风速(m/s)', '地表气压(kPa)']
    st.dataframe(df_show, use_container_width=True)
except Exception as e:
    st.warning(f"未找到示例数据：{e}")

# ==================== 加载标准化参数 ====================
@st.cache_data
def load_scaler():
    mean = np.load("data/scaler_mean.npy")
    std = np.load("data/scaler_scale.npy")
    return mean, std

mean_vals, std_vals = load_scaler()

# ==================== 第二部分：在线预测 ====================
st.subheader("2️⃣ 在线预测")
st.markdown("输入过去 **7天** 的气象数据，模型将预测 **第8天** 的平均气温。")

default_data = pd.DataFrame({
    '平均气温(℃)': [28.0, 29.5, 30.2, 31.0, 29.8, 28.5, 27.0],
    '最高气温(℃)': [33.0, 34.5, 35.2, 36.0, 34.8, 33.5, 32.0],
    '最低气温(℃)': [23.0, 24.5, 25.2, 26.0, 24.8, 23.5, 22.0],
    '相对湿度(%)': [65.0, 62.0, 60.0, 58.0, 63.0, 68.0, 72.0],
    '风速(m/s)': [2.5, 2.8, 3.0, 2.2, 2.6, 3.2, 2.9],
    '地表气压(kPa)': [100.5, 100.3, 100.1, 100.0, 100.4, 100.6, 100.8],
})
edited_df = st.data_editor(default_data, num_rows="fixed", use_container_width=True)

if st.button("🔮 开始预测", type="primary"):
    try:
        input_data = edited_df.values
        # 1. 标准化输入
        input_normalized = (input_data - mean_vals) / std_vals
        input_tensor = torch.FloatTensor(input_normalized).unsqueeze(0)
        
        with torch.no_grad():
            pred_normalized = model(input_tensor).item()
            
        # 2. 反标准化输出，还原为摄氏度
        pred_temp = pred_normalized * std_vals[0] + mean_vals[0]
        st.success(f"### 🌡️ 预测第8天平均气温：**{pred_temp:.2f} ℃**")
    except Exception as e:
        st.error(f"预测失败：{e}")

# ==================== 第三部分：模型预测结果对比图 ====================
st.subheader("3️⃣ 模型预测结果对比图")
st.markdown("点击下方按钮，实时运行模型推理并生成对比图：")

output_dir = "output"
X = np.load("data/X.npy")
y = np.load("data/y.npy")
split = int(len(X) * 0.8)
X_test = X[split:]
y_test = y[split:]
X_test_t = torch.FloatTensor(X_test)

def draw_and_show(show_extreme=False):
    """加载两个模型，画三条线，在内存中渲染"""
    # 加载基础模型
    base_model = TempConvLSTM()
    base_model.load_state_dict(torch.load("output/convlstm_model.pth", map_location='cpu'))
    base_model.eval()
    
    # 加载物理约束模型
    physics_model = TempConvLSTM()
    physics_model.load_state_dict(torch.load("output/convlstm_physics_model.pth", map_location='cpu'))
    physics_model.eval()
    
    with torch.no_grad():
        y_pred_base = base_model(X_test_t).numpy()
        y_pred_physics = physics_model(X_test_t).numpy()
        
    n = 200
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(y_test[:n], label='真实气温 (Ground Truth)', color='blue', linewidth=2, alpha=0.7)
    ax.plot(y_pred_base[:n], label='纯数据ConvLSTM预测', color='red', linestyle='--', linewidth=1.5)
    ax.plot(y_pred_physics[:n], label='物理约束ConvLSTM预测', color='green', linestyle='-.', linewidth=1.5)
    
    if show_extreme:
        extreme_idx = np.where(y_test[:n] > 1.0)[0]
        if len(extreme_idx) > 0:
            ax.scatter(extreme_idx, y_test[extreme_idx], color='orange', s=40, zorder=5, label='极端高温样本')
            
    ax.set_title('ConvLSTM Temperature Prediction')
    ax.set_xlabel('Test Sample Index')
    ax.set_ylabel('Normalized Temperature')
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)
    
    mse_base = np.mean((y_test - y_pred_base)**2)
    mse_physics = np.mean((y_test - y_pred_physics)**2)
    st.success(f"✅ 推理完成！纯数据模型 MSE = {mse_base:.4f} | 物理约束模型 MSE = {mse_physics:.4f}")

col1, col2 = st.columns(2)
with col1:
    if st.button("📊 运行并查看整体对比图"):
        st.session_state.run_normal = True
with col2:
    if st.button("🌡️ 运行并查看极端天气图"):
        st.session_state.run_extreme = True

if st.session_state.get("run_normal"):
    draw_and_show(show_extreme=False)
if st.session_state.get("run_extreme"):
    draw_and_show(show_extreme=True)
# ==================== 第四部分：极端天气表现 ====================
st.subheader("4️⃣ 极端天气表现（答辩重点）")
st.markdown("""
- **普通天气**：纯模型精度略高（MSE 0.0413）
- **物理约束模型**：牺牲微小精度（MSE 0.0419），换取极端天气下的物理鲁棒性。
- **极端高温样本**：单独切出气温 > 35℃ 的样本进行验证，MSE 从 0.0205 降至 0.0157。
""")

st.divider()
st.caption(f"© 2026 南信大数统院挑战杯项目组 - 相风 | 更新：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
