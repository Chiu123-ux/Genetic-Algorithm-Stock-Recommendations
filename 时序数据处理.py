import pandas as pd
import pickle
import re  # 新增正则库

# 加载数据
with open("C:/Users/bu'zhi'h'h'h/Desktop/pycharm/遗传算法/gplearnplus/use-gplearn-to-generate-CTA-factor/factor_data.pkl", 'rb') as f:
    original_data = pickle.load(f)

target_dict = {}

for i, factor_name in enumerate(original_data['factor_names']):
    # 从因子名称中解析窗口大小（例如"turnover_21"提取21）
    window_size = int(re.search(r'\d+$', factor_name).group())  # 关键修改1

    # 提取原始三维数据
    factor_data = original_data['factors'][:, :, i]  # 形状 (1211, 3495)

    # 创建原始DataFrame
    df = pd.DataFrame(
        data=factor_data,
        index=pd.to_datetime(original_data['dates']),
        columns=original_data['stocks']
    )

    # 执行滚动等权平均（允许部分窗口计算）
    df_rolling = df.rolling(window=window_size, min_periods=1).mean()  # 关键修改2

    # 保留最后window_size个原始数据（覆盖滚动计算结果）
    df_rolling.iloc[-window_size:] = df.iloc[-window_size:]  # 关键修改3

    target_dict[factor_name] = df_rolling

print(target_dict)

with open("factor_time.pkl", 'wb') as f:
     pickle.dump(target_dict, f)
print("字典已保存为 factor_time.pkl")