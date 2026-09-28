import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import plotly.express as px
from datetime import datetime
from convlstm import ConvLSTM

st.set_page_config(page_title="相风 - 南京微气候预测", layout="wide")
st.title("🌬️ 相风 - 南京微气候预测平台")

st.markdown("""
> **项目介绍**：本项目基于ConvLSTM及物理约束，对南京地区历史气象数据（2015-2025）进行学习，实现未来短时气温预测与可视化。
""")

# ==================== 加载模型和标准化参数 ====================
@st.cache_resource
def load_model():
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

    model = TempConvLSTM()
    model.load_state_dict(torch.load("output/convlstm_physics_model.pth", map_location='cpu'))
    model.eval()
    return model

@st.cache_data
def load_scaler_params():
    """从原始CSV计算标准化参数，避免依赖sklearn"""
    df = pd.read_csv("data/nanjing.csv", skiprows=17)
    features = ['T2M', 'T2M_MAX', 'T2M_MIN', 'RH2M', 'WS2M', 'PS']
    mean = df[features].mean().values
    std = df[features].std().values
    return mean, std

model = load_model()
mean_vals, std_vals = load_scaler_params()

# ==================== 第一部分：数据预览 ====================
st.subheader("1️⃣ 数据预览")
st.markdown("以下是南京地区历史气象数据的示例（每行代表一天）：")

try:
    df_demo = pd.read_csv("data/nanjing.csv", skiprows=17)
    df_show = df_demo.head(10).copy()
    df_show.columns = ['年份', '年积日', '平均气温(℃)', '最高气温(℃)', '最低气温(℃)',
                       '相对湿度(%)', '风速(m/s)', '地表气压(kPa)']
    st.dataframe(df_show, use_container_width=True)
except Exception as e:
    st.warning(f"未找到本地示例数据：{e}")

# ==================== 第二部分：在线预测 ====================
st.subheader("2️⃣ 在线预测")
st.markdown("输入过去 **7天** 的气象数据，模型将预测 **第8天** 的平均气温。")

st.markdown("**请输入过去7天的数据（每行一天，共7行）：**")

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
        input_data = edited_df.values  # shape (7, 6)
        input_normalized = (input_data - mean_vals) / std_vals
        input_tensor = torch.FloatTensor(input_normalized).unsqueeze(0)  # (1, 7, 6)

        with torch.no_grad():
            pred_normalized = model(input_tensor).item()

        pred_temp = pred_normalized * std_vals[0] + mean_vals[0]
        st.success(f"### 🌡️ 预测第8天平均气温：**{pred_temp:.2f} ℃**")
    except Exception as e:
        st.error(f"预测失败：{e}")

# ==================== 第三部分：模型预测结果对比图 ====================
st.subheader("3️⃣ 模型预测结果对比图")
st.markdown("点击下方按钮，切换查看不同实验的对比图：")

output_dir = "output"

def load_img_from_txt(txt_name):
    txt_path = os.path.join(output_dir, txt_name)
    if not os.path.exists(txt_path):
        return None
    with open(txt_path, "r", encoding="utf-8") as f:
        img_name = f.read().strip()
    img_path = os.path.join(output_dir, img_name)
    return img_path if os.path.exists(img_path) else None

col1, col2, col3 = st.columns(3)
with col1:
    btn_normal = st.button("📊 查看整体对比图")
with col2:
    btn_extreme = st.button("🌡️ 查看极端天气图")
with col3:
    btn_both = st.button("📋 显示全部")

if "view_mode" not in st.session_state:
    st.session_state.view_mode = "both"
if btn_normal:
    st.session_state.view_mode = "normal"
if btn_extreme:
    st.session_state.view_mode = "extreme"
if btn_both:
    st.session_state.view_mode = "both"

mode = st.session_state.view_mode

if mode in ("normal", "both"):
    img = load_img_from_txt("latest_normal.txt")
    if img:
        st.image(img, caption="📊 整体模型预测对比图（纯ConvLSTM vs 物理约束）", use_container_width=True)
    else:
        st.info("尚未生成整体对比图。")

if mode in ("extreme", "both"):
    img = load_img_from_txt("latest_extreme.txt")
    if img:
        st.image(img, caption="🌡️ 极端天气高亮对比图（MSE降低23.4%）", use_container_width=True)
    else:
        st.info("尚未生成极端天气高亮图。")

# ==================== 第四部分：极端天气表现 ====================
st.subheader("4️⃣ 极端天气表现（答辩重点）")
st.markdown("""
- **普通天气**：纯模型精度略高（MSE 0.0413）
- **物理约束模型**：牺牲微小精度（MSE 0.0419），换取了极端天气下的物理鲁棒性。
- **极端高温样本**：我们单独切出了气温 > 35℃ 的样本进行验证。
""")

st.divider()
st.caption(f"© 2026 南信大数统院挑战杯项目组 - 相风 | 更新时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
