import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

csv_path = "D:/Weather_Project/data/nanjing.csv"

# 1. 自动寻找表头所在的行
with open(csv_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

header_row = 0
for i, line in enumerate(lines):
    # 找到包含 YEAR 和 DOY 的那一行，就是真正的表头
    if 'YEAR' in line and 'DOY' in line:
        header_row = i
        break

print(f"检测到表头在第 {header_row + 1} 行，开始读取...")

# 2. 读取数据（skiprows=header_row 意味着跳过表头前面的说明文字，把表头那行当做列名）
df = pd.read_csv(csv_path, skiprows=header_row)

print("列名:", df.columns.tolist())
print("数据量:", len(df), "行")

# 3. 合并日期
df['date'] = pd.to_datetime(df['YEAR'].astype(str) + '-' + df['DOY'].astype(str), format='%Y-%j')
df = df.sort_values('date').reset_index(drop=True)

# 4. 处理缺失值（NASA用-999表示缺失，替换成NaN后用线性插值填补）
df = df.replace(-999, np.nan)
df = df.interpolate(method='linear')

# 5. 标准化
features = ['T2M', 'T2M_MAX', 'T2M_MIN', 'RH2M', 'WS2M', 'PS']
scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])

np.save("data/scaler_mean.npy", scaler.mean_)
np.save("data/scaler_scale.npy", scaler.scale_)
# 6. 构建滑动窗口（输入过去7天，预测未来1天气温）
def create_sequences(data, input_len=7, pred_len=1):
    X, y = [], []
    for i in range(len(data) - input_len - pred_len + 1):
        X.append(data[i : i + input_len])
        y.append(data[i + input_len : i + input_len + pred_len, :].squeeze(0))
    return np.array(X), np.array(y)

arr = df[features].values
X, y = create_sequences(arr, input_len=7, pred_len=1)

# 形状校验
assert X.shape[1:] == (7, 6), f"X形状错误: {X.shape}"
assert y.shape[1:] == (6,), f"y形状错误: {y.shape}"

print("输入形状:", X.shape, "输出形状:", y.shape)
np.save("D:/Weather_Project/data/X.npy", X)
np.save("D:/Weather_Project/data/y.npy", y)
print("✅ 数据预处理完成！")