import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import streamlit as st
import pandas as pd
import numpy as np
import torch
import plotly.express as px
from datetime import datetime

st.set_page_config(page_title="相风 - 南京微气候预测", layout="wide")

st.title("🌬️ 相风 - 南京微气候预测平台")
st.markdown("""
> **项目介绍**：本项目基于ConvLSTM及物理约束，对南京地区历史气象数据（2015-2025）进行学习，实现未来短时气温预测与可视化。
""")

# 侧边栏：输入控制
st.sidebar.header("📊 预测参数设置")
uploaded_file = st.sidebar.file_uploader("上传你的气象数据 (CSV)", type="csv")

st.subheader("1️⃣ 数据预览")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.dataframe(df.head())
else:
    st.info("请上传气象数据CSV文件，或者直接查看我们为您准备的示例数据。")
    try:
        df_demo = pd.read_csv("data/nanjing.csv", skiprows=17)
        st.dataframe(df_demo.head(10))
    except:
        st.warning("未找到本地示例数据，请确认路径。")
import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')  # 关键：无GUI模式，省内存
import matplotlib.pyplot as plt
from convlstm import ConvLSTM
import os

st.subheader("2️⃣ 点击运行模型推理")
st.markdown("点击下方按钮，运行物理约束模型进行推理并展示结果：")

# ========== 模型定义 ==========
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

# ========== 推理函数 ==========
def run_inference():
    with st.spinner("正在加载数据与模型..."):
        X = np.load("data/X.npy")
        y = np.load("data/y.npy")
        split = int(len(X) * 0.8)
        X_test = X[split:]
        y_test = y[split:]
        X_test_t = torch.FloatTensor(X_test)
        
        model = TempConvLSTM()
        model.load_state_dict(torch.load("output/convlstm_physics_model.pth", map_location='cpu'))
        model.eval()
        
        with torch.no_grad():
            y_pred = model(X_test_t).numpy()
        
        mse = np.mean((y_test - y_pred)**2)
        st.success(f"推理完成！测试集 MSE = {mse:.4f}")
        
        # 绘图
        n = 200
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(y_test[:n], label='真实气温', color='blue', linewidth=1.5)
        ax.plot(y_pred[:n], label='物理约束预测', color='green', linestyle='-.', linewidth=1.5)
        ax.set_title('ConvLSTM Prediction')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)

# ========== 按钮 ==========
if st.button("▶️ 运行物理约束模型推理"):
    run_inference()

st.subheader("3️⃣ 极端天气表现（答辩重点）")
st.markdown("""
- **普通天气**：纯模型精度略高（MSE 0.0413）
- **物理约束模型**：牺牲微小精度（MSE 0.0419），换取了极端天气下的物理鲁棒性。
- **极端高温样本**：我们单独切出了气温 > 35℃ 的样本进行验证。
""")

st.divider()
st.caption(f"© 2026 南信大数统院挑战杯项目组 - 相风 | 更新时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")