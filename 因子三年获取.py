import tushare as ts
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import dill
import time
from datetime import datetime, timedelta

# 设置 Tushare Token
ts.set_token('bacbf02a717535992d2e41973171eaddef76803fea0fbd9ae50acf3c')
pro = ts.pro_api()

# ========== 配置参数 ==========
YEARS = 3
MAX_RETRIES = 3
REQ_INTERVAL = 0.4
BATCH_SIZE = 50  # 每批处理的股票数量
FIELDS = ['open', 'close', 'vol', 'amount', 'change']  # 需要的字段

# ========== 工具函数 ==========
def get_valid_stocks():
    """获取非ST且上市满5年的股票列表"""
    df = pro.stock_basic(exchange='', list_status='L', fields='ts_code,name,list_date')
    cutoff_date = (datetime.now() - timedelta(days=365 * YEARS)).strftime('%Y%m%d')
    df = df[~df['name'].str.contains('ST') & (df['list_date'] < cutoff_date)]
    return df['ts_code'].tolist()

def fetch_single_stock(ts_code, start_date, end_date):
    """获取单只股票的指定字段数据"""
    for attempt in range(MAX_RETRIES):
        try:
            df = pro.daily(
                ts_code=ts_code,
                start_date=start_date,
                end_date=end_date,
                fields='trade_date,open,high,low,close,pre_close,change,pct_chg,vol,amount'
            )
            if df.empty:
                print(f"股票 {ts_code} 无数据")
                return None

            df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
            df = df.set_index('trade_date').sort_index()
            df.index.name = None  # 去除索引名称

            # 提取所需字段并重命名列
            result = {field: df[[field]] for field in FIELDS}
            return {field: result[field].rename(columns={field: ts_code}) for field in result}
        except Exception as e:
            print(f"股票 {ts_code} 获取失败 (第{attempt+1}次重试): {str(e)}")
            time.sleep(5)
    return None

# ========== 主流程 ==========
if __name__ == '__main__':
    # 获取有效股票列表
    valid_stocks = get_valid_stocks()
    print(f"共 {len(valid_stocks)} 只有效股票")

    # 时间范围
    end_date = datetime.now().strftime('%Y%m%d')
    start_date = (datetime.now() - timedelta(days=365 * YEARS)).strftime('%Y%m%d')

    # 初始化各因子DataFrame
    factor_data = {field: pd.DataFrame() for field in FIELDS}

    # 分批获取数据
    for i in tqdm(range(0, len(valid_stocks), BATCH_SIZE), desc="获取数据进度"):
        batch = valid_stocks[i:i + BATCH_SIZE]

        with ThreadPoolExecutor(max_workers=6) as executor:
            futures = [executor.submit(fetch_single_stock, code, start_date, end_date) for code in batch]

            for future in futures:
                stock_data = future.result()
                if stock_data:
                    for field in FIELDS:
                        factor_data[field] = pd.concat([factor_data[field], stock_data[field]], axis=1)

        # 批次间隔避免限速
        if i + BATCH_SIZE < len(valid_stocks):
            time.sleep(30)

    # 数据清洗：填充缺失值、排序
    for field in FIELDS:
        factor_data[field] = factor_data[field].ffill().bfill().sort_index()
        print(f"{field} 数据维度: {factor_data[field].shape}")

    # 保存为 .pkl 文件
    with open('factor_3years.pkl', 'wb') as f:
        dill.dump(factor_data, f)
    print("✅ 因子数据已保存为 factor_3years.pkl")