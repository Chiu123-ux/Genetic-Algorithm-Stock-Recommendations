import tushare as ts
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import dill
import time
from datetime import datetime, timedelta

# 设置 Tushare Token
ts.set_token('bacbf02a717535992d2e41973171eaddef76803fea0fbd9ae50acf3c')
pro = ts.pro_api()

# ========== 配置参数 ==========
YEARS = 1
MAX_RETRIES = 3
BATCH_SIZE = 30

# 精选必要因子 - 确保稳定获取且对选股有效
ESSENTIAL_FACTORS = {
    'close': '收盘价',
    'open': '开盘价',
    'high': '最高价', 
    'low': '最低价',
    'pct_chg': '涨跌幅',
    'vol': '成交量',
    'amount': '成交额',
    'pe': '市盈率',
    'pb': '市净率',
    'turnover_rate': '换手率',
    'total_mv': '总市值',
    'circ_mv': '流通市值'
}

# ========== 工具函数 ==========
def get_valid_stocks():
    """获取非ST且上市满指定年限的股票列表"""
    try:
        df = pro.stock_basic(exchange='', list_status='L', 
                           fields='ts_code,name,list_date,market')
        
        cutoff_date = (datetime.now() - timedelta(days=365 * YEARS)).strftime('%Y%m%d')
        
        # 过滤ST股票和上市时间不足的股票
        df = df[~df['name'].str.contains('ST|退', na=False) & 
                (df['list_date'] < cutoff_date)]
        
        print(f"获取到 {len(df)} 只有效股票")
        return df['ts_code'].tolist()
    
    except Exception as e:
        print(f"获取股票列表失败: {str(e)}")
        return []

def fetch_single_stock_data(ts_code, start_date, end_date):
    """获取单只股票的所有因子数据，返回一个DataFrame"""
    for attempt in range(MAX_RETRIES):
        try:
            result_data = {}
            
            # 1. 获取日线行情数据
            df_daily = pro.daily(
                ts_code=ts_code,
                start_date=start_date,
                end_date=end_date
            )
            
            if df_daily.empty:
                return None
            
            # 处理日线数据
            df_daily['trade_date'] = pd.to_datetime(df_daily['trade_date'], format='%Y%m%d')
            df_daily = df_daily.set_index('trade_date').sort_index()

            # 提取基础因子
            for field in ['open', 'high', 'low', 'close', 'pct_chg', 'vol', 'amount']:
                if field in df_daily.columns:
                    result_data[field] = df_daily[field]

            # 2. 获取每日指标数据
            try:
                df_basic = pro.daily_basic(
                    ts_code=ts_code, 
                    start_date=start_date, 
                    end_date=end_date
                )
                
                if not df_basic.empty:
                    df_basic['trade_date'] = pd.to_datetime(df_basic['trade_date'], format='%Y%m%d')
                    df_basic = df_basic.set_index('trade_date').sort_index()
                    
                    for field in ['pe', 'pb', 'turnover_rate', 'total_mv', 'circ_mv']:
                        if field in df_basic.columns:
                            result_data[field] = df_basic[field]
                            
            except Exception as e:
                print(f"获取 {ts_code} 估值数据失败: {str(e)}")

            # 合并所有数据
            if result_data:
                # 创建多列DataFrame，每列是一个因子
                stock_df = pd.DataFrame(result_data)
                stock_df.columns = [f"{col}_{ts_code}" for col in stock_df.columns]
                return stock_df
            else:
                return None

        except Exception as e:
            print(f"股票 {ts_code} 获取失败 (第{attempt+1}次重试): {str(e)}")
            time.sleep(2)
    
    return None

def reorganize_data_by_factor(all_stock_data):
    """将数据重新组织为按因子分类的DataFrame字典"""
    print("重新组织数据格式...")
    
    # 初始化每个因子的DataFrame
    factor_dataframes = {}
    
    # 收集所有日期
    all_dates = set()
    for stock_df in all_stock_data.values():
        if stock_df is not None:
            all_dates.update(stock_df.index)
    
    if not all_dates:
        return {}
    
    all_dates = sorted(all_dates)
    
    # 为每个因子创建空的DataFrame
    base_factors = ['close', 'open', 'high', 'low', 'pct_chg', 'vol', 'amount', 
                   'pe', 'pb', 'turnover_rate', 'total_mv', 'circ_mv']
    
    for factor in base_factors:
        factor_dataframes[factor] = pd.DataFrame(index=all_dates)
    
    # 填充数据
    for ts_code, stock_df in tqdm(all_stock_data.items(), desc="组织数据"):
        if stock_df is not None:
            for factor in base_factors:
                factor_col = f"{factor}_{ts_code}"
                if factor_col in stock_df.columns:
                    factor_dataframes[factor][ts_code] = stock_df[factor_col]
    
    return factor_dataframes

def calculate_derived_factors(factor_dataframes):
    """计算衍生因子"""
    print("计算衍生技术指标...")
    
    if 'close' not in factor_dataframes:
        return factor_dataframes
    
    # 计算移动平均线
    for window in [5, 10, 20]:
        ma_name = f'MA{window}'
        factor_dataframes[ma_name] = factor_dataframes['close'].rolling(window=window).mean()
    
    # 计算收益率动量
    factor_dataframes['RETURN_5'] = factor_dataframes['close'].pct_change(5)
    factor_dataframes['RETURN_10'] = factor_dataframes['close'].pct_change(10)
    factor_dataframes['RETURN_20'] = factor_dataframes['close'].pct_change(20)
    
    # 计算波动率
    factor_dataframes['VOLATILITY_20'] = factor_dataframes['close'].pct_change().rolling(20).std()
    
    # 计算成交量均线
    if 'vol' in factor_dataframes:
        factor_dataframes['VOLUME_MA5'] = factor_dataframes['vol'].rolling(5).mean()
        factor_dataframes['VOLUME_MA20'] = factor_dataframes['vol'].rolling(20).mean()
    
    return factor_dataframes

# ========== 主流程 ==========
if __name__ == '__main__':
    print("开始获取选股因子数据...")
    print("=" * 50)
    
    # 显示将要获取的因子
    print("核心因子列表:")
    for i, (field, desc) in enumerate(ESSENTIAL_FACTORS.items(), 1):
        print(f"  {i:2d}. {field:15s} - {desc}")
    print("=" * 50)
    
    # 获取有效股票列表
    valid_stocks = get_valid_stocks()
    if not valid_stocks:
        print("未获取到有效股票列表，程序退出")
        exit()
    
    # 时间范围
    end_date = datetime.now().strftime('%Y%m%d')
    start_date = (datetime.now() - timedelta(days=365 * YEARS)).strftime('%Y%m%d')
    print(f"时间范围: {start_date} 至 {end_date}")
    print(f"股票数量: {len(valid_stocks)}")
    
    # 存储所有股票数据
    all_stock_data = {}
    successful_stocks = []
    
    # 分批获取数据
    total_batches = (len(valid_stocks) + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"\n开始分批获取数据，共 {total_batches} 个批次...")
    
    for i in tqdm(range(0, len(valid_stocks), BATCH_SIZE), desc="总体进度", total=total_batches):
        batch = valid_stocks[i:i + BATCH_SIZE]
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(fetch_single_stock_data, code, start_date, end_date): code for code in batch}
            
            for future in tqdm(futures, desc=f"批次 {i//BATCH_SIZE + 1}", leave=False):
                stock_data = future.result()
                stock_code = futures[future]
                if stock_data is not None:
                    all_stock_data[stock_code] = stock_data
                    successful_stocks.append(stock_code)
        
        # 批次间隔避免限速
        if i + BATCH_SIZE < len(valid_stocks):
            print("等待2秒避免请求限制...")
            time.sleep(2)
    
    print(f"\n数据获取完成，成功获取 {len(successful_stocks)} 只股票数据")
    
    # 重新组织数据格式
    factor_dataframes = reorganize_data_by_factor(all_stock_data)
    
    if not factor_dataframes:
        print("没有有效数据，程序退出")
        exit()
    
    # 计算衍生因子
    factor_dataframes = calculate_derived_factors(factor_dataframes)
    
    # 数据清洗
    print("数据清洗中...")
    for factor_name, df in factor_dataframes.items():
        # 向前填充然后向后填充
        factor_dataframes[factor_name] = df.ffill().bfill()
        print(f"{factor_name}: {df.shape}")
    
    # 最终统计
    print("\n" + "=" * 50)
    print("最终数据统计:")
    print(f"📊 总因子数量: {len(factor_dataframes)}")
    print(f"📈 总股票数量: {len(successful_stocks)}")
    
    if 'close' in factor_dataframes:
        total_dates = len(factor_dataframes['close'].index)
        date_range_start = factor_dataframes['close'].index.min().strftime('%Y-%m-%d')
        date_range_end = factor_dataframes['close'].index.max().strftime('%Y-%m-%d')
        print(f"📅 总交易日数: {total_dates}")
        print(f"📅 数据时间范围: {date_range_start} 至 {date_range_end}")
    
    print("\n可用因子列表:")
    derived_factors = ['MA5', 'MA10', 'MA20', 'RETURN_5', 'RETURN_10', 'RETURN_20', 'VOLATILITY_20', 'VOLUME_MA5', 'VOLUME_MA20']
    base_factors = list(ESSENTIAL_FACTORS.keys())
    
    all_factors = base_factors + derived_factors
    available_factors = [f for f in all_factors if f in factor_dataframes]
    
    for i, factor in enumerate(available_factors, 1):
        desc = ESSENTIAL_FACTORS.get(factor, '衍生技术指标')
        shape = factor_dataframes[factor].shape
        print(f"  {i:2d}. {factor:15s} - {desc} {shape}")
    
    # 保存为pkl文件 - 与你原来格式一致
    output_file = f'stock_factors_{YEARS}years.pkl'
    
    # 可以选择保存单个因子或多个因子
    # 保存所有因子到一个字典中
    save_data = {
        'factors': factor_dataframes,
        'metadata': {
            'stock_count': len(successful_stocks),
            'date_range': f"{date_range_start} 至 {date_range_end}",
            'factor_count': len(factor_dataframes),
            'creation_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    }
    
    with open(output_file, 'wb') as f:
        dill.dump(save_data, f)
    
    # 同时保存一个单独的收盘价文件，方便直接使用
    close_output_file = f'close_price_{YEARS}years.pkl'
    with open(close_output_file, 'wb') as f:
        dill.dump(factor_dataframes['close'], f)
    
    print(f"\n✅ 数据保存完成!")
    print(f"📁 主文件: {output_file} (包含所有{len(factor_dataframes)}个因子)")
    print(f"📁 收盘价文件: {close_output_file} (单独收盘价数据)")
    print(f"💡 数据格式与您原来的pkl文件一致")
    
    # 显示数据格式示例
    if 'close' in factor_dataframes:
        close_df = factor_dataframes['close']
        print(f"\n📊 数据格式示例 (close因子):")
        print(f"   类型: {type(close_df)}")
        print(f"   形状: {close_df.shape}")
        print(f"   索引: {close_df.index.name}")
        print(f"   列数: {len(close_df.columns)}")
        print(f"   前5只股票: {list(close_df.columns[:5])}")
        print(f"\n   前3行数据示例:")
        print(close_df.head(3).iloc[:, :5])  # 显示前3行前5列