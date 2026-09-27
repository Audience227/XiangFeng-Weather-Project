import numpy as np
import pandas as pd
import torch
import xarray as xr

print("Numpy:", np.__version__)
print("Pandas:", pd.__version__)
print("PyTorch:", torch.__version__)
print("xarray:", xr.__version__)
print("CUDA可用:",
torch.cuda.is_available())
print("环境配置完全成功")