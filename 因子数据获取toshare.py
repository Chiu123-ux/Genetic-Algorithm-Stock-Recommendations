import tushare as ts
import pandas as pd
import numpy as np
import dill
import time
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor

# 初始化Tushare
ts.set_token('')
pro = ts.pro_api()

# ========== 配置参数 ==========
FACTOR_FIELDS = {
    'turnover_21': 'turnover_rate',
    'avg_volume_63': 'volume_ratio',
    'amount_21': 'total_mv',
    'turnover_63': 'float_share',
    'turnover_252': 'circ_mv'
}
YEARS = 3
MAX_RETRIES = 3
REQ_INTERVAL = 0.4

# ========== 工具函数 ==========
def get_valid_stocks():  # 添加缺失的函数定义
    """获取非ST且上市满5年的股票列表"""
    df = pro.stock_basic(exchange='', list_status='L', fields='ts_code,name,list_date')
    cutoff_date = (datetime.now() - timedelta(days=365 * YEARS)).strftime('%Y%m%d')
    df = df[~df['name'].str.contains('ST') & (df['list_date'] < cutoff_date)]
    return df['ts_code'].tolist()

def get_single_stock_data(ts_code, start_date, end_date):
    """获取单只股票价格数据（含重试）"""
    for attempt in range(MAX_RETRIES):
        try:
            df = pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date, adj='qfq')
            df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
            df = df[['trade_date', 'close']].drop_duplicates('trade_date').set_index('trade_date')
            df.columns = [ts_code]
            time.sleep(REQ_INTERVAL)
            return df
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                print(f"股票 {ts_code} 第{attempt+1}次重试...")
                time.sleep(5)
                continue
            print(f"股票 {ts_code} 价格获取失败: {str(e)}")
            return pd.DataFrame()

def get_single_factor(ts_code, field, start_date, end_date):
    """获取单只股票单个因子数据（含重试）"""
    for attempt in range(MAX_RETRIES):
        try:
            df = pro.daily_basic(ts_code=ts_code, start_date=start_date, end_date=end_date, fields='trade_date,' + field)
            df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
            df = df.set_index('trade_date')[field].rename(ts_code)
            time.sleep(REQ_INTERVAL)
            return df
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                print(f"因子 {field} 第{attempt+1}次重试...")
                time.sleep(5)
                continue
            print(f"股票 {ts_code} 因子 {field} 获取失败: {str(e)}")
            return pd.Series()

# ========== 主流程 ==========
if __name__ == '__main__':
    # 步骤1: 获取股票列表
    valid_stocks = get_valid_stocks()  # 现在可以正常调用
    print(f"有效股票数量: {len(valid_stocks)}")

    # 步骤2: 配置时间范围
    end_date = datetime.now().strftime('%Y%m%d')
    start_date = (datetime.now() - timedelta(days=365 * YEARS)).strftime('%Y%m%d')

    # 步骤3: 分批获取股票价格数据（每50只一组）
    print("\n====== 正在获取股票价格数据 ======")
    BATCH_SIZE = 50  # 新增：分批大小
    price_data = pd.DataFrame()

    for batch_idx in range(0, len(valid_stocks), BATCH_SIZE):
        batch_stocks = valid_stocks[batch_idx:batch_idx + BATCH_SIZE]

        with ThreadPoolExecutor(max_workers=6) as executor:  # 降低并发数
            futures = [executor.submit(get_single_stock_data, code, start_date, end_date) for code in batch_stocks]
            batch_df = pd.concat([f.result() for f in futures if not f.result().empty], axis=1)
            price_data = pd.concat([price_data, batch_df], axis=1)

        print(f"已完成批次 {batch_idx // BATCH_SIZE + 1}/{(len(valid_stocks) // BATCH_SIZE) + 1}")
        if batch_idx + BATCH_SIZE < len(valid_stocks):
            time.sleep(30)  # 新增：批次间隔

    # 数据清洗（保持不变）
    price_data = price_data.sort_index().ffill().dropna(axis=1, how='all')
    price_data.to_pickle('stock_prices.pkl')
    print(f"价格数据维度: {price_data.shape}")

    # ========== 步骤4: 分批获取因子数据（优化版） ==========
    print("\n====== 正在获取因子数据 ======")
    factor_dict = {}

    for factor, tushare_field in FACTOR_FIELDS.items():
        print(f"\n=== 处理因子: {factor} ===")
        factor_df = pd.DataFrame(index=price_data.index)

        for batch_idx in range(0, len(valid_stocks), BATCH_SIZE):
            batch_stocks = valid_stocks[batch_idx:batch_idx + BATCH_SIZE]
            print(f"当前批次股票: {batch_stocks[:3]}...")  # 显示前3只股票便于调试

            with ThreadPoolExecutor(max_workers=4) as executor:
                # 提交任务
                futures = [executor.submit(get_single_factor, code, tushare_field, start_date, end_date) for code in
                           batch_stocks]

                # 收集非空结果
                batch_data = pd.DataFrame()
                for future in futures:
                    try:
                        result = future.result()
                        if not result.empty:
                            batch_data = pd.concat([batch_data, result], axis=1)
                    except Exception as e:
                        print(f"严重错误: {str(e)}")

                # 合并批次数据（仅当有数据时）
                if not batch_data.empty:
                    factor_df = pd.concat([factor_df, batch_data], axis=1)
                    print(f"因子 {factor} 批次 {batch_idx // BATCH_SIZE + 1} 有效股票数: {batch_data.shape[1]}")
                else:
                    print(f"警告: 因子 {factor} 批次 {batch_idx // BATCH_SIZE + 1} 无有效数据")

            # 频率控制
            if batch_idx + BATCH_SIZE < len(valid_stocks):
                time.sleep(30)

        # 最终处理
        if factor_df.empty:
            print(f"错误: 因子 {factor} 无任何数据，请检查字段映射或接口权限")
            continue

        factor_dict[factor] = factor_df.ffill().bfill()
        print(f"因子 {factor} 总有效股票数: {factor_df.shape[1]}")

    # 后续处理保持不变
    factor_array = np.stack([factor_dict[f].values for f in FACTOR_FIELDS.keys()], axis=-1)

    with open('factor_data.pkl', 'wb') as f:
        dill.dump({
            'factors': factor_array,
            'dates': price_data.index,
            'stocks': price_data.columns.tolist(),
            'factor_names': list(FACTOR_FIELDS.keys())
        }, f)
    print(f"因子数据维度: {factor_array.shape}")


print("\n====== 全部数据已保存 ======")
