import tushare as ts
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 设置 Tushare Token（已填写你的token）
ts.set_token('bacbf02a717535992d2e41973171eaddef76803fea0fbd9ae50acf3c')
pro = ts.pro_api()


# ========== 步骤1：获取全A股列表并剔除ST ==========
def get_all_a_stocks():
    # 获取全A股基础信息（含上市状态）
    df = pro.stock_basic(
        exchange='',
        list_status='L',  # L表示上市状态
        fields='ts_code,name,list_date'  # 免费版不提供ST字段，仅获取必要字段
    )
    # 剔除名称中含"ST"的股票（免费版替代方案）
    df = df[~df['name'].str.contains('ST')]
    # 排除上市不足5年的股票（确保数据覆盖周期）
    cutoff_date = (datetime.now() - timedelta(days=365 * 5)).strftime('%Y%m%d')
    df = df[df['list_date'] < cutoff_date]
    return df['ts_code'].tolist()


valid_stocks = get_all_a_stocks()

# ========== 步骤2：批量获取股票历史数据（时间跨度3年） ==========
end_date = datetime.now().strftime('%Y%m%d')
start_date = (datetime.now() - timedelta(days=365 * 3)).strftime('%Y%m%d')

# 定义数据收集器（日期为索引，股票代码为列名）
all_data = pd.DataFrame()

for idx, ts_code in enumerate(valid_stocks):
    try:
        # 获取前复权日线数据
        df = pro.daily(
            ts_code=ts_code,
            start_date=start_date,
            end_date=end_date,
            adj='qfq'  # 前复权
        )[['trade_date', 'close']]

        # 转换日期格式并去重
        df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
        df = df.drop_duplicates(subset='trade_date').set_index('trade_date')
        df.columns = [ts_code]  # 列名改为股票代码

        # 合并数据（高效方式）
        all_data = pd.concat([all_data, df], axis=1)

        # 进度提示（每50只输出一次）
        if (idx + 1) % 50 == 0:
            print(f"已获取 {idx + 1}/{len(valid_stocks)} 只股票数据")

    except Exception as e:
        print(f"股票 {ts_code} 获取失败: {str(e)}")

# ========== 步骤3：数据清洗 ==========
# 按日期排序
all_data = all_data.sort_index()
# 填充缺失值（用前一日数据填充）
all_data = all_data.ffill()
# 剔除全为NaN的列（无效股票）
all_data = all_data.dropna(axis=1, how='all')

# ========== 步骤4：保存数据 ==========
all_data.to_pickle('stock_data_3years.pkl')  # 修改文件名以反映过滤ST
print(f"数据已保存，维度：{all_data.shape}")

