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

st.subheader("2️⃣ 模型预测结果对比图")
st.markdown("下图展示了纯ConvLSTM模型与物理约束ConvLSTM模型在测试集上的预测效果对比：")

# 找到最新的对比图
output_dir = "output"
png_files = [f for f in os.listdir(output_dir) if f.startswith('三条线对比') and f.endswith('.png')]

if png_files:
    latest_png = max(png_files, key=lambda x: os.path.getctime(os.path.join(output_dir, x)))
    st.image(os.path.join(output_dir, latest_png), caption=f"最近一次预测对比：{latest_png}", use_container_width=True)
else:
    st.warning("⚠️ 未找到对比图，请先在命令行运行 python evaluate.py 生成图片。")

st.subheader("3️⃣ 极端天气表现（答辩重点）")
st.markdown("""
- **普通天气**：纯模型精度略高（MSE 0.0413）
- **物理约束模型**：牺牲微小精度（MSE 0.0419），换取了极端天气下的物理鲁棒性。
- **极端高温样本**：我们单独切出了气温 > 35℃ 的样本进行验证。
""")

st.divider()
st.caption(f"© 2026 南信大数统院挑战杯项目组 - 相风 | 更新时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")