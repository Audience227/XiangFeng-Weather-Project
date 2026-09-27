import numpy as np
X = np.load("D:/Weather_Project/data/X.npy")
y = np.load("D:/Weather_Project/data/y.npy")
print("X 的形状：", X.shape)
print("y 的形状：", y.shape)
print("看看 X 的第一组数据（过去7天）：\n", X[0])
print("对应的 y 的答案（未来1天）：", y[0])