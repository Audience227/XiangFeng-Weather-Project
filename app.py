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
st.markdown("下图展示了纯ConvLSTM模型与物理约束ConvLSTM模型的预测效果对比：")

import os, time

output_dir = "output"
time_normal_file = os.path.join(output_dir, "time_normal.txt")
time_extreme_file = os.path.join(output_dir, "time_extreme.txt")

t1 = None
t2 = None

if os.path.exists(time_normal_file):
    with open(time_normal_file, "r") as f:
        t1 = float(f.read().strip())
if os.path.exists(time_extreme_file):
    with open(time_extreme_file, "r") as f:
        t2 = float(f.read().strip())

# 时间窗口：90秒（1.5分钟）。若两个脚本运行时间相差在此之内，视为同一次实验，都展示。
THRESHOLD = 90

show_normal = False
show_extreme = False

if t1 and t2:
    if abs(t1 - t2) < THRESHOLD:
        # 两个脚本几乎同时运行过，都展示
        show_normal = True
        show_extreme = True
    else:
        # 只展示最近运行过的那个
        if t1 > t2:
            show_normal = True
        else:
            show_extreme = True
elif t1:
    show_normal = True
elif t2:
    show_extreme = True

# 展示普通对比图
if show_normal:
    normal_txt = os.path.join(output_dir, "latest_normal.txt")
    if os.path.exists(normal_txt):
        with open(normal_txt, "r", encoding="utf-8") as f:
            name = f.read().strip()
        img_path = os.path.join(output_dir, name)
        if os.path.exists(img_path):
            st.image(img_path, caption="整体模型预测对比图", use_container_width=True)

# 展示极端天气图
if show_extreme:
    extreme_txt = os.path.join(output_dir, "latest_extreme.txt")
    if os.path.exists(extreme_txt):
        with open(extreme_txt, "r", encoding="utf-8") as f:
            name = f.read().strip()
        img_path = os.path.join(output_dir, name)
        if os.path.exists(img_path):
            st.image(img_path, caption="极端天气高亮对比图", use_container_width=True)

if not show_normal and not show_extreme:
    st.warning("⚠️ 未找到对比图，请先运行 evaluate.py 或 evaluate2.py 生成图片。")
st.subheader("3️⃣ 极端天气表现（答辩重点）")
st.markdown("""
- **普通天气**：纯模型精度略高（MSE 0.0413）
- **物理约束模型**：牺牲微小精度（MSE 0.0419），换取了极端天气下的物理鲁棒性。
- **极端高温样本**：我们单独切出了气温 > 35℃ 的样本进行验证。
""")

st.divider()
st.caption(f"© 2026 南信大数统院挑战杯项目组 - 相风 | 更新时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")