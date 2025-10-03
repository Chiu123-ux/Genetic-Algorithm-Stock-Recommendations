import tushare as ts
import pandas as pd
import os
import time
from datetime import datetime, timedelta

# 设置 token
ts.set_token('')
pro = ts.pro_api()

# 获取A股代码列表
stock_info = pro.query('stock_basic', exchange='', list_status='L', fields='ts_code,symbol,name,area,industry,list_date')
a_stock_codes = stock_info[stock_info['ts_code'].str.endswith(('.SH', '.SZ'))]['ts_code'].tolist()

# 定义时间范围
start_date = '20180501'
end_date = '20250501'

# 分块获取数据
def fetch_in_chunks(ts_code, start_date, end_date, chunk_days=300):
    current_start = start_date
    all_df = pd.DataFrame()
    while current_start <= end_date:
        current_end = (datetime.strptime(current_start, "%Y%m%d") + timedelta(days=chunk_days)).strftime("%Y%m%d")
        if current_end > end_date:
            current_end = end_date
        df = pro.daily(ts_code=ts_code, start_date=current_start, end_date=current_end)
        if not df.empty:
            all_df = pd.concat([all_df, df], ignore_index=True)
        current_start = (datetime.strptime(current_end, "%Y%m%d") + timedelta(days=1)).strftime("%Y%m%d")
        time.sleep(0.1)
    return all_df

# 批量下载
def fetch_all_a_stock_data(stock_codes, start_date, end_date, save_path='data/2018-2025'):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    total = len(stock_codes)
    for i, ts_code in enumerate(stock_codes):
        print(f"[{i+1}/{total}] Fetching {ts_code}...")
        df = fetch_in_chunks(ts_code, start_date, end_date)
        if not df.empty:
            df.to_csv(f"{save_path}/{ts_code.replace('.', '_')}.csv", index=False)
        else:
            print(f"{ts_code} no data")
        time.sleep(0.1)

# 执行

fetch_all_a_stock_data(a_stock_codes, start_date, end_date)
