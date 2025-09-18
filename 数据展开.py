import pickle
import pandas as pd
# try:
# 加载pickle文件
with open("C:/Users/bu'zhi'h'h'h/Desktop/pycharm/遗传算法/gplearnplus/use-gplearn-to-generate-CTA-factor/data/factor_data.pickle", 'rb') as file:
        data = pickle.load(file)

for key, value in data.items():
    print(f"特征 {key} 的形状:", value.shape)

#         # 转换逻辑
#         if isinstance(data, list):
#             if all(isinstance(item, dict) for item in data):  # 字典列表
#                 df = pd.DataFrame(data)
#             else:  # 普通列表
#                 df = pd.DataFrame(data, columns=['列1', '列2'])
#         elif isinstance(data, dict):  # 单层字典
#             df = pd.DataFrame([data])
#         else:
#             raise ValueError("不支持的数据类型")
#
#     # 保存Excel
#     df.to_csv('output.csv', index=False)
#     print("转换成功！")
# except Exception as e:
#     print(f"错误：{e}")

# import pandas as pd
# data2 = pd.read_pickle('stock_data_5years_fullA_noST.pkl')
# print(data2.head())

