import tushare as ts
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import dill
import time
from datetime import datetime, timedelta

# 设置 Tushare Token
ts.set_token('你的token')
pro = ts.pro_api()

# ========== 配置参数 ==========
YEARS = 1
MAX_RETRIES = 3
BATCH_SIZE = 30
MAX_STOCKS = 800

# 精选必要因子
ESSENTIAL_FACTORS = {
    'close': '收盘价',
    'open': '开盘价',
    'high': '最高价', 
    'low': '最低价',
    'pct_chg': '涨跌幅',
    'vol': '成交量',
    'amount': '成交额',
    'turnover_rate': '换手率',
    'total_mv': '总市值',
    'circ_mv': '流通市值'
}

# ========== 工具函数 ==========
def get_valid_stocks_by_liquidity(max_stocks=800):
    """获取流动性最好的股票列表"""
    try:
        # 获取基础股票信息
        df = pro.stock_basic(exchange='', list_status='L', 
                           fields='ts_code,name,list_date,market')
        
        # 计算截止日期
        cutoff_date = (datetime.now() - timedelta(days=365)).strftime('%Y%m%d')
        
        # 过滤ST股票和上市时间不足的股票
        df = df[~df['name'].str.contains('ST|退', na=False) & 
                (df['list_date'] < cutoff_date)]
        
        # 获取最近一天的流动性数据来排序
        trade_date = (datetime.now() - timedelta(days=5)).strftime('%Y%m%d')
        
        try:
            market_data = pro.daily_basic(trade_date=trade_date, 
                                        fields='ts_code,total_mv,turnover_rate,circ_mv')
            
            # 合并数据
            df = df.merge(market_data, on='ts_code', how='left')
            df = df.dropna(subset=['total_mv', 'turnover_rate'])
            df['liquidity_score'] = df['circ_mv'] * df['turnover_rate']
            df = df.sort_values('liquidity_score', ascending=False)
            
            selected_stocks = df.head(max_stocks)['ts_code'].tolist()
            print(f"✅ 基于流动性筛选出 {len(selected_stocks)} 只股票")
            return selected_stocks
            
        except:
            print("⚠️  使用备用股票选择方案")
            selected_stocks = df.head(max_stocks)['ts_code'].tolist()
            return selected_stocks
    
    except Exception as e:
        print(f"获取股票列表失败: {str(e)}")
        return []

def fetch_single_stock_data(ts_code, start_date, end_date):
    """获取单只股票的所有因子数据"""
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
                    
                    for field in ['turnover_rate', 'total_mv', 'circ_mv']:
                        if field in df_basic.columns:
                            result_data[field] = df_basic[field]
                            
            except Exception as e:
                print(f"获取 {ts_code} 估值数据失败: {str(e)}")

            # 合并所有数据
            if result_data:
                stock_df = pd.DataFrame(result_data)
                stock_df.columns = [f"{col}_{ts_code}" for col in stock_df.columns]
                return stock_df
            else:
                return None

        except Exception as e:
            print(f"股票 {ts_code} 获取失败 (第{attempt+1}次重试): {str(e)}")
            time.sleep(2)
    
    return None



def calculate_volume_amplification_indicators(factor_dataframes):
    """计算成交量放大指标 - 修复pct_change警告"""
    print("📈 计算成交量放大指标...")
    
    if 'vol' not in factor_dataframes:
        return factor_dataframes
    
    volume_data = factor_dataframes['vol']
    
    # 核心成交量放大因子
    factor_dataframes['VOLUME_RATIO_5'] = volume_data / volume_data.rolling(5).mean()
    factor_dataframes['VOLUME_RATIO_10'] = volume_data / volume_data.rolling(10).mean()
    factor_dataframes['VOLUME_RATIO_20'] = volume_data / volume_data.rolling(20).mean()
    
    # 成交量突破信号
    factor_dataframes['VOLUME_BREAK_5'] = volume_data > volume_data.rolling(5).mean() * 1.5
    factor_dataframes['VOLUME_BREAK_10'] = volume_data > volume_data.rolling(10).mean() * 1.8
    factor_dataframes['VOLUME_BREAK_20'] = volume_data > volume_data.rolling(20).mean() * 2.0
    
    # 成交量动量
    factor_dataframes['VOLUME_MOMENTUM_5'] = volume_data / volume_data.shift(5)
    factor_dataframes['VOLUME_MOMENTUM_10'] = volume_data / volume_data.shift(10)
    
    # 量价齐升指标 - 修复pct_change警告
    if 'close' in factor_dataframes:
        close_data = factor_dataframes['close'].fillna(method='ffill').fillna(method='bfill')
        
        price_change_5 = close_data.pct_change(5, fill_method=None)
        volume_change_5 = volume_data / volume_data.shift(5)
        factor_dataframes['PRICE_VOLUME_SYNC_5'] = (price_change_5 > 0) & (volume_change_5 > 1.5)
        
        price_change_10 = close_data.pct_change(10, fill_method=None)
        volume_change_10 = volume_data / volume_data.shift(10)
        factor_dataframes['PRICE_VOLUME_SYNC_10'] = (price_change_10 > 0.05) & (volume_change_10 > 1.8)
    
    print("✅ 成交量放大指标计算完成")
    return factor_dataframes

def clean_and_save_data(factor_dataframes, successful_stocks):
    """数据清洗和保存"""
    print("数据清洗中...")
    
    # 统一的缺失值处理
    for factor_name in factor_dataframes:
        df = factor_dataframes[factor_name]
        # 先向前填充，再向后填充，最后用0填充剩余缺失值
        df_cleaned = df.ffill().bfill().fillna(0)
        factor_dataframes[factor_name] = df_cleaned
        print(f"{factor_name}: {df_cleaned.shape}")
    
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
    
    # 显示成交量相关因子
    volume_factors = [f for f in factor_dataframes.keys() if 'VOLUME' in f]
    print(f"\n🎯 成交量放大因子 ({len(volume_factors)}个):")
    for i, factor in enumerate(volume_factors, 1):
        print(f"  {i:2d}. {factor}")
    
    # 保存为pkl文件
    output_file = f'stock_factors_{MAX_STOCKS}stocks_1year.pkl'
    
    save_data = {
        'factors': factor_dataframes,
        'metadata': {
            'stock_count': len(successful_stocks),
            'date_range': f"{date_range_start} 至 {date_range_end}",
            'factor_count': len(factor_dataframes),
            'volume_factors': volume_factors,
            'creation_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    }
    
    with open(output_file, 'wb') as f:
        dill.dump(save_data, f)
    
    # 同时保存单独的收盘价文件
    close_output_file = f'close_price_{MAX_STOCKS}stocks_1year.pkl'
    with open(close_output_file, 'wb') as f:
        dill.dump(factor_dataframes['close'], f)
    
    print(f"\n✅ 数据保存完成!")
    print(f"📁 主文件: {output_file}")
    print(f"📁 收盘价文件: {close_output_file}")
    print(f"🎯 特别包含 {len(volume_factors)} 个成交量放大因子")

# ========== 主流程 ==========
if __name__ == '__main__':
    print("开始获取选股因子数据...")
    print("=" * 50)
    
    # 获取流动性最好的800只股票
    valid_stocks = get_valid_stocks_by_liquidity(MAX_STOCKS)
    if not valid_stocks:
        print("未获取到有效股票列表，程序退出")
        exit()
    
    # 时间范围
    end_date = datetime.now().strftime('%Y%m%d')
    start_date = (datetime.now() - timedelta(days=365 * 1)).strftime('%Y%m%d')
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
            print("等待10秒避免请求限制...")
            time.sleep(10)
    
    print(f"\n数据获取完成，成功获取 {len(successful_stocks)} 只股票数据")
    
    # 重新组织数据格式
    factor_dataframes = reorganize_data_by_factor(all_stock_data)
    
    if not factor_dataframes:
        print("没有有效数据，程序退出")
        exit()
    
    # 计算技术指标
    factor_dataframes = calculate_technical_indicators(factor_dataframes)
    
    # 计算成交量放大指标
    factor_dataframes = calculate_volume_amplification_indicators(factor_dataframes)
    
    # 数据清洗和保存

    clean_and_save_data(factor_dataframes, successful_stocks)

