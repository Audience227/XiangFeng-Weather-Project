from pynasapower.get_data import query_power
from pynasapower.geometry import point
import datetime

# 南京地区坐标
gpoint = point(118.7, 32.2, "EPSG:4326")
start = datetime.date(2015, 1, 1)
end = datetime.date(2025, 12, 31)

# 下载数据
data = query_power(
    geometry=gpoint,
    start=start,
    end=end,
    community="ag",
    parameters=["T2M", "T2M_MAX", "T2M_MIN", "RH2M", "WS2M", "PS"],
    temporal_api="daily",
    format="csv",
    path="D:/Weather_Project/data/"
)
print("数据下载完成！")