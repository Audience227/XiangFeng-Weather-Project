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
from zoneinfo import ZoneInfo
from convlstm import ConvLSTM

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

st.set_page_config(page_title="相风 XiangFeng | 南京微气候预测平台", layout="wide", page_icon="🌬️")

# ==================== 商务视觉层 ====================
CUSTOM_CSS = """
<style>
  html, body, [class*="css"] { font-family: "Microsoft YaHei", "PingFang SC", "Segoe UI", sans-serif; }
  /* 主视觉横幅 */
  .hero { background: linear-gradient(135deg, #0F2A43 0%, #1E5799 55%, #2E86DE 100%);
          padding: 2.4rem 2.6rem 2.1rem; border-radius: 18px; color: #FFFFFF;
          box-shadow: 0 6px 24px rgba(15, 42, 67, .28); }
  .hero h1 { font-size: 2.7rem; font-weight: 800; margin: 0; letter-spacing: 3px; }
  .hero .en { font-size: .95rem; letter-spacing: 2.5px; opacity: .75; text-transform: uppercase; margin-top: .3rem; }
  .hero .sub { font-size: 1.08rem; opacity: .93; margin-top: .9rem; line-height: 1.6; }
  .badge { display: inline-block; background: rgba(255,255,255,.13); border: 1px solid rgba(255,255,255,.28);
           padding: .28rem 1rem; border-radius: 999px; font-size: .82rem; margin: 1rem .5rem 0 0; }
  /* 章节标题 */
  .sec { display: flex; align-items: center; gap: .75rem; margin: 1.8rem 0 .5rem; }
  .sec .num { background: linear-gradient(135deg, #1F4E79, #2E86DE); color: #fff; font-weight: 700;
              border-radius: 9px; padding: .18rem .68rem; font-size: 1.02rem; letter-spacing: 1px; }
  .sec .txt { font-size: 1.42rem; font-weight: 800; color: #16283C; letter-spacing: 1px; }
  .sec .line { flex: 1; height: 1px; background: linear-gradient(90deg, #C9D6E4, transparent); }
  /* 按钮与卡片 */
  .stButton > button { background: linear-gradient(135deg, #1F4E79, #2E86DE); color: #fff; border: none;
                       border-radius: 10px; padding: .52rem 1.5rem; font-weight: 700; letter-spacing: 1px;
                       box-shadow: 0 3px 10px rgba(31, 78, 121, .28); transition: all .2s; }
  .stButton > button:hover { filter: brightness(1.12); transform: translateY(-1px); }
  [data-testid="stMetric"] { background: #F7F9FC; border: 1px solid #E4EAF2; border-radius: 14px;
                             padding: 1.05rem 1.2rem; box-shadow: 0 2px 6px rgba(22, 40, 60, .05); }
  [data-testid="stMetricValue"] { color: #1F4E79; font-weight: 800; }
  [data-testid="stMetricLabel"] { color: #5A6B7E; font-weight: 600; }
  /* 侧边栏 */
  [data-testid="stSidebar"] { background: linear-gradient(180deg, #0F2A43, #16395C); }
  [data-testid="stSidebar"] * { color: #DCE6F2 !important; }
  [data-testid="stSidebar"] hr { border-color: rgba(255,255,255,.18); }
  .side-tag { background: rgba(255,255,255,.10); border: 1px solid rgba(255,255,255,.22);
              border-radius: 9px; padding: .5rem .8rem; font-size: .85rem; margin-bottom: .6rem; line-height: 1.5; }
  /* 数据框与页脚 */
  [data-testid="stDataFrame"] { border-radius: 12px; }
  .footer { text-align: center; color: #7A8A9C; font-size: .85rem; padding: 1.2rem 0 .4rem;
            border-top: 1px solid #E4EAF2; margin-top: 2rem; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ==================== 侧边栏：项目档案 ====================
with st.sidebar:
    st.markdown("### 🌬️ 相风 XiangFeng")
    st.markdown('<div class="side-tag"><b>数据来源</b><br>NASA POWER 官方公开数据集<br>南京 32.20°N, 118.70°E<br>2015–2025 · 逐日 · 4011 条</div>', unsafe_allow_html=True)
    st.markdown('<div class="side-tag"><b>模型架构</b><br>ConvLSTM (hidden=16)<br>输入：过去 7 天 × 6 要素<br>输出：第 8 天 6 项气象指标</div>', unsafe_allow_html=True)
    st.markdown('<div class="side-tag"><b>对比实验</b><br>纯数据 vs 物理约束<br>MSE 口径：无单位（标准化空间）<br>高温日判定：气象局标准</div>', unsafe_allow_html=True)
    st.markdown('<div class="side-tag"><b>团队</b><br>南信大数统院 "数学+AI" 方向<br>挑战杯 · 科技作品</div>', unsafe_allow_html=True)

# ==================== 主视觉 ====================
st.markdown("""
<div class="hero">
  <h1>相 风</h1>
  <div class="en">XiangFeng · Microclimate Forecasting Platform</div>
  <div class="sub">基于 ConvLSTM 与物理约束的南京微气候预测可视化平台 —— 用现代技术回答古人千年之问</div>
  <span class="badge">🛰️ NASA POWER 数据源</span><span class="badge">📅 2015–2025 十年观测</span><span class="badge">🧠 ConvLSTM + 物理约束</span><span class="badge">🌡️ 6 项气象指标预测</span>
</div>
""", unsafe_allow_html=True)

# ==================== 模型定义 ====================
class TempConvLSTM(nn.Module):
    """六输出模型：预测6项气象指标（在线预测用）"""
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

class TempConvLSTM1D(nn.Module):
    """单输出模型：只预测气温（对比实验用，与文稿/evaluate2.py同结构同权重）"""
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

# ==================== 加载模型和标准化参数（只加载一次） ====================
@st.cache_resource
def load_all_models():
    # 六输出模型（在线预测）
    model6 = TempConvLSTM()
    model6.load_state_dict(torch.load("output/best_convlstm_model.pth", map_location='cpu'))
    model6.eval()
    # 单输出模型（对比图，与文稿实验口径一致）
    base1 = TempConvLSTM1D()
    base1.load_state_dict(torch.load("output/convlstm_model.pth", map_location='cpu'))
    base1.eval()
    physics1 = TempConvLSTM1D()
    physics1.load_state_dict(torch.load("output/convlstm_physics_model.pth", map_location='cpu'))
    physics1.eval()
    return model6, base1, physics1

@st.cache_data
def load_scaler():
    mean = np.load("data/scaler_mean.npy")
    std = np.load("data/scaler_scale.npy")
    return mean, std

@st.cache_data
def load_test_data():
    X = np.load("data/X.npy")
    y = np.load("data/y.npy")
    split = int(len(X) * 0.8)
    return X[split:], y[split:]

model6, base_model, physics_model = load_all_models()
mean_vals, std_vals = load_scaler()

X_test, y_test = load_test_data()
# 真实单位（℃）：日均温（预测主对象）与日最高气温（用于气象局高温日判定）
y_test_temp = y_test[:, 0] * std_vals[0] + mean_vals[0]
y_test_tmax = y_test[:, 1] * std_vals[1] + mean_vals[1]

# 高温日判定：中国气象局标准——日最高气温 ≥ 35℃ 记为高温日
# （注意：判定用"日最高气温"T2M_MAX，模型评估的是"日平均气温"预测）
EXTREME_TMAX_C = 35.0

# 一次性推理并缓存预测结果（避免每次交互都重跑模型）
@st.cache_data
def get_predictions(_X_test):
    """返回 (℃单位预测, 标准化单位预测)；单输出模型，与文稿实验同口径"""
    X_t = torch.FloatTensor(_X_test)
    with torch.no_grad():
        pred_base_n = base_model(X_t).numpy()[:, 0]
        pred_physics_n = physics_model(X_t).numpy()[:, 0]
    return (pred_base_n * std_vals[0] + mean_vals[0],
            pred_physics_n * std_vals[0] + mean_vals[0],
            pred_base_n, pred_physics_n)

y_pred_base, y_pred_physics, pred_base_n, pred_physics_n = get_predictions(X_test)

# ==================== 核心指标总览 ====================
heat_idx_all = np.where(y_test_tmax >= EXTREME_TMAX_C)[0]
rmse_all_base = float(np.sqrt(np.mean((y_test_temp - y_pred_base)**2)))
rmse_all_phys = float(np.sqrt(np.mean((y_test_temp - y_pred_physics)**2)))
mse_heat_base = float(np.mean((y_test[heat_idx_all, 0] - pred_base_n[heat_idx_all])**2))
mse_heat_phys = float(np.mean((y_test[heat_idx_all, 0] - pred_physics_n[heat_idx_all])**2))
rmse_heat_base = float(np.sqrt(np.mean((y_test_temp[heat_idx_all] - y_pred_base[heat_idx_all])**2)))
rmse_heat_phys = float(np.sqrt(np.mean((y_test_temp[heat_idx_all] - y_pred_physics[heat_idx_all])**2)))
drop_pct = (1 - mse_heat_phys / mse_heat_base) * 100

m1, m2, m3, m4 = st.columns(4)
m1.metric("整体预测精度 RMSE", f"{rmse_all_phys:.2f} ℃", f"纯数据模型 {rmse_all_base:.2f} ℃", delta_color="off")
m2.metric("高温日 RMSE", f"{rmse_heat_phys:.2f} ℃", f"纯数据模型 {rmse_heat_base:.2f} ℃", delta_color="off")
m3.metric("高温日误差下降", f"-{drop_pct:.1f}%", "物理约束的关键贡献", delta_color="off")
m4.metric("测试集高温日", f"{len(heat_idx_all)} 天", "日最高气温 ≥ 35℃", delta_color="off")

# ==================== 第一部分：数据预览 ====================
st.markdown('<div class="sec"><span class="num">01</span><span class="txt">数据预览</span><span class="line"></span></div>', unsafe_allow_html=True)
try:
    df_demo = pd.read_csv("data/nanjing.csv", skiprows=17)
    df_show = df_demo.head(10).copy()
    df_show.columns = ['年份', '年积日', '平均气温(℃)', '最高气温(℃)', '最低气温(℃)',
                       '相对湿度(%)', '风速(m/s)', '地表气压(kPa)']
    st.dataframe(df_show, use_container_width=True)
except Exception as e:
    st.warning(f"未找到示例数据：{e}")

# ==================== 第二部分：在线预测 ====================
st.markdown('<div class="sec"><span class="num">02</span><span class="txt">在线预测</span><span class="line"></span></div>', unsafe_allow_html=True)
st.markdown("输入过去 **7天** 的气象数据，模型将预测 **第8天** 的6项气象指标。")

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
        input_normalized = (input_data - mean_vals) / std_vals
        input_tensor = torch.FloatTensor(input_normalized).unsqueeze(0)

        with torch.no_grad():
            pred_normalized = model6(input_tensor).numpy().flatten()

        pred_original = pred_normalized * std_vals + mean_vals

        feature_names = ['平均气温(℃)', '最高气温(℃)', '最低气温(℃)',
                         '相对湿度(%)', '风速(m/s)', '地表气压(kPa)']

        result_df = pd.DataFrame({
            '气象指标': feature_names,
            '预测值': np.round(pred_original, 2)
        })

        st.success("### 🌡️ 第8天预测结果：")
        st.dataframe(result_df, use_container_width=True, hide_index=True)
    except Exception as e:
        st.error(f"预测失败：{e}")

# ==================== 第三部分：对比图（真实℃单位） ====================
st.markdown('<div class="sec"><span class="num">03</span><span class="txt">模型预测结果对比</span><span class="line"></span></div>', unsafe_allow_html=True)
st.markdown("点击下方按钮，切换查看对比图（坐标单位：℃ 真实值；所用权重与文稿实验一致）：")

def draw_and_show(show_extreme=False):
    n = 200
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(y_test_temp[:n], label='Ground Truth', color='blue', linewidth=2, alpha=0.7)
    ax.plot(y_pred_base[:n], label='Pure ConvLSTM', color='red', linestyle='--', linewidth=1.5)
    ax.plot(y_pred_physics[:n], label='Physics-constrained ConvLSTM', color='green', linestyle='-.', linewidth=1.5)

    if show_extreme:
        heat_idx_all = np.where(y_test_tmax >= EXTREME_TMAX_C)[0]
        plot_heat_idx = [i for i in heat_idx_all if i < n]
        if len(plot_heat_idx) > 0:
            ax.scatter(plot_heat_idx, y_test_temp[plot_heat_idx], color='orange',
                       s=40, zorder=5, label='CMA heat days (T2M_MAX ≥ 35°C)')

    ax.set_title('Daily Mean Temperature Prediction (Nanjing)')
    ax.set_xlabel('Test Sample Index (Day)')
    ax.set_ylabel('Temperature (℃)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)

    # MSE 统一采用无单位（标准化）口径，与答辩材料一致；RMSE 换算为实际温度作直觉参考
    if show_extreme:
        heat_idx = np.where(y_test_tmax >= EXTREME_TMAX_C)[0]
        if len(heat_idx) > 0:
            mse_base_n = np.mean((y_test[heat_idx, 0] - pred_base_n[heat_idx])**2)
            mse_physics_n = np.mean((y_test[heat_idx, 0] - pred_physics_n[heat_idx])**2)
            rmse_base_c = np.sqrt(np.mean((y_test_temp[heat_idx] - y_pred_base[heat_idx])**2))
            rmse_physics_c = np.sqrt(np.mean((y_test_temp[heat_idx] - y_pred_physics[heat_idx])**2))
            drop = (1 - mse_physics_n / mse_base_n) * 100
            st.success(
                f"✅ **高温日推理完成！**（气象局标准：日最高气温 ≥ {EXTREME_TMAX_C:.0f}℃，测试集共 **{len(heat_idx)}** 天）\n\n"
                f"- 纯数据模型 MSE = **{mse_base_n:.4f}**（无单位，标准化口径）| 对应 RMSE ≈ **{rmse_base_c:.2f} ℃**\n"
                f"- 物理约束模型 MSE = **{mse_physics_n:.4f}**（无单位，标准化口径）| 对应 RMSE ≈ **{rmse_physics_c:.2f} ℃**\n"
                f"- 高温日误差相对下降 **-{drop:.1f}%**"
            )
        else:
            st.warning("测试集中未发现高温日样本。")
    else:
        mse_base_n = np.mean((y_test[:, 0] - pred_base_n)**2)
        mse_physics_n = np.mean((y_test[:, 0] - pred_physics_n)**2)
        rmse_base_c = np.sqrt(np.mean((y_test_temp - y_pred_base)**2))
        rmse_physics_c = np.sqrt(np.mean((y_test_temp - y_pred_physics)**2))
        st.success(
            f"✅ **整体推理完成！**\n\n"
            f"- 纯数据模型 MSE = **{mse_base_n:.4f}**（无单位，标准化口径）| 对应 RMSE ≈ **{rmse_base_c:.2f} ℃**\n"
            f"- 物理约束模型 MSE = **{mse_physics_n:.4f}**（无单位，标准化口径）| 对应 RMSE ≈ **{rmse_physics_c:.2f} ℃**"
        )

col1, col2 = st.columns(2)
with col1:
    if st.button("📊 查看整体对比图"):
        st.session_state.view_mode = "normal"
with col2:
    if st.button("🌡️ 查看极端天气图"):
        st.session_state.view_mode = "extreme"

if "view_mode" not in st.session_state:
    st.session_state.view_mode = "normal"

if st.session_state.view_mode == "normal":
    draw_and_show(show_extreme=False)
elif st.session_state.view_mode == "extreme":
    draw_and_show(show_extreme=True)

# ==================== 第四部分：极端天气表现 ====================
st.markdown('<div class="sec"><span class="num">04</span><span class="txt">极端天气表现</span><span class="line"></span></div>', unsafe_allow_html=True)
st.markdown(f"""
- **高温日判定标准**：采用中国气象局标准——**日最高气温 ≥ {EXTREME_TMAX_C:.0f}℃** 记为高温日；
  测试集（约 2024–2025 年，803 天）中共 **38 个高温日**（全十年 4011 天共 177 个）。
  注意：判定用日最高气温（T2M_MAX），模型评估的是日平均气温（T2M）的预测。
- **指标口径**：MSE 为**无单位（标准化空间）**指标，与答辩材料/消融实验口径一致；
  同时附换算后的实际温度 RMSE（℃）便于直观理解。
- **实验结论**：物理约束模型整体精度与纯数据模型相当，在**高温日上预测误差显著更低**，
  体现物理约束带来的鲁棒性提升。具体数值点击上方按钮实时查看。
""")

st.markdown(f"""
<div class="footer">© 2026 南京信息工程大学 数统院挑战杯项目组 · 相风 XiangFeng<br>
数据来源：NASA POWER | 更新时间：{datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y-%m-%d %H:%M')}（北京时间）</div>
""", unsafe_allow_html=True)
