import tushare as ts
import pickle
import pandas as pd
'''
# 初始化Tushare
ts.set_token('')
pro = ts.pro_api()
print(pro.query('daily_basic', fields='', src=''))
# 查询daily_basic接口所有可用字段
print("Tushare daily_basic接口字段:", pro.query('daily_basic', fields='').columns.tolist())
'''

with open(
        "C:/Users/bu'zhi'h'h'h/Desktop/pycharm/遗传算法/gplearnplus/use-gplearn-to-generate-CTA-factor/factor_data.pkl",
        'rb') as f:
    original_data = pickle.load(f)

target_dict = {}

for i, factor_name in enumerate(original_data['factor_names']):
    # 正确提取因子数据：形状应为 (1211, 3495)
    factor_data = original_data['factors'][:, :, i]  # 关键修改：使用 [:, :, i] 而不是 [i, :, :]

    # 创建 DataFrame
    df = pd.DataFrame(
        data=factor_data,
        index=original_data['dates'],  # 1211 行
        columns=original_data['stocks']  # 3495 列
    )

    target_dict[factor_name] = df

print(target_dict)

# 保存字典到 pkl 文件
# with open("factor_test.pkl", 'wb') as f:
#     pickle.dump(target_dict, f)
# print("字典已保存为 factor_test.pkl")

with open(
        "C:/Users/bu'zhi'h'h'h/Desktop/pycharm/遗传算法/gplearnplus/use-gplearn-to-generate-CTA-factor/stock_prices.pkl",
        'rb') as f:
    data = pickle.load(f)

print(data)
