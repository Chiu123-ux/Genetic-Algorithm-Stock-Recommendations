import tushare as ts
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import dill
import time
from datetime import datetime, timedelta
import os
import gc

# 设置 Tushare Token
ts.set_token('bacbf02a717535992d2e41973171eaddef76803fea0fbd9ae50acf3c')
pro = ts.pro_api()

# ========== 配置参数 ==========
YEARS = 3  # 改为3年
MAX_RETRIES = 5  # 增加重试次数
BATCH_SIZE = 20  # 减小批次大小，避免内存问题
MAX_STOCKS = 800  # 减少股票数量，3年数据量更大

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
    """获取流动性最好的股票列表 - 修复版本"""
    try:
        print("🔄 获取股票列表...")
        # 获取基础股票信息
        df = pro.stock_basic(exchange='', list_status='L', 
                           fields='ts_code,name,list_date,market')
        
        print(f"初始获取股票数量: {len(df)}")
        
        # 过滤ST股票
        df = df[~df['name'].str.contains('ST|退', na=False)]
        print(f"过滤ST后股票数量: {len(df)}")
        
        # 计算截止日期 - 3年前
        cutoff_date = (datetime.now() - timedelta(days=365*3)).strftime('%Y%m%d')
        
        # 过滤上市时间不足的股票
        df = df[df['list_date'] < cutoff_date]
        print(f"过滤上市时间后股票数量: {len(df)}")
        
        if len(df) == 0:
            print("❌ 过滤后没有符合条件的股票")
            return []
        
        # 尝试获取市场数据来排序
        try:
            print("尝试获取市场数据...")
            # 获取最近5个交易日的数据，选择有数据的那天
            recent_date = (datetime.now() - timedelta(days=5)).strftime('%Y%m%d')
            
            market_data = pro.daily_basic(
                trade_date=recent_date, 
                fields='ts_code,total_mv,turnover_rate,circ_mv'
            )
            
            if market_data.empty:
                # 如果当天没有数据，尝试前一天
                recent_date = (datetime.now() - timedelta(days=6)).strftime('%Y%m%d')
                market_data = pro.daily_basic(
                    trade_date=recent_date, 
                    fields='ts_code,total_mv,turnover_rate,circ_mv'
                )
            
            if not market_data.empty:
                print(f"获取到 {len(market_data)} 只股票的市场数据")
                
                # 合并数据
                df = df.merge(market_data, on='ts_code', how='inner')  # 使用inner join确保有市场数据
                df = df.dropna(subset=['total_mv', 'turnover_rate'])
                
                if len(df) > 0:
                    df['liquidity_score'] = df['circ_mv'] * df['turnover_rate']
                    df = df.sort_values('liquidity_score', ascending=False)
                    
                    selected_stocks = df.head(max_stocks)['ts_code'].tolist()
                    print(f"✅ 基于流动性筛选出 {len(selected_stocks)} 只股票")
                    return selected_stocks
                else:
                    print("❌ 合并市场数据后没有有效股票")
            else:
                print("❌ 无法获取市场数据")
                
        except Exception as e:
            print(f"获取市场数据失败: {e}")
        
        # 备用方案：按上市时间排序（最新上市的在前）
        print("🔄 使用备用方案：按上市时间排序")
        df = df.sort_values('list_date', ascending=False)
        selected_stocks = df.head(max_stocks)['ts_code'].tolist()
        print(f"✅ 备用方案筛选出 {len(selected_stocks)} 只股票")
        return selected_stocks
    
    except Exception as e:
        print(f"❌ 获取股票列表失败: {str(e)}")
        return []

def fetch_single_stock_data(ts_code, start_date, end_date):
    """获取单只股票的所有因子数据 - 修复版本"""
    for attempt in range(MAX_RETRIES):
        try:
            result_data = {}
            
            # 1. 获取日线行情数据 - 添加更长的延迟
            time.sleep(0.2)  # 添加延迟避免请求过快
            
            df_daily = pro.daily(
                ts_code=ts_code,
                start_date=start_date,
                end_date=end_date
            )
            
            if df_daily.empty:
                print(f"⚠️  股票 {ts_code} 无日线数据")
                return None
            
            # 处理日线数据
            df_daily['trade_date'] = pd.to_datetime(df_daily['trade_date'], format='%Y%m%d')
            df_daily = df_daily.set_index('trade_date').sort_index()

            # 检查数据量是否足够
            if len(df_daily) < 100:  # 至少100个交易日的数据
                print(f"⚠️  股票 {ts_code} 数据量不足: {len(df_daily)} 天")
                return None

            # 提取基础因子
            for field in ['open', 'high', 'low', 'close', 'pct_chg', 'vol', 'amount']:
                if field in df_daily.columns:
                    result_data[field] = df_daily[field]

            # 2. 获取每日指标数据 - 添加错误处理
            try:
                time.sleep(0.1)  # 添加延迟
                df_basic = pro.daily_basic(
                    ts_code=ts_code, 
                    start_date=start_date, 
                    end_date=end_date,
                    fields='ts_code,trade_date,turnover_rate,total_mv,circ_mv'
                )
                
                if not df_basic.empty:
                    df_basic['trade_date'] = pd.to_datetime(df_basic['trade_date'], format='%Y%m%d')
                    df_basic = df_basic.set_index('trade_date').sort_index()
                    
                    for field in ['turnover_rate', 'total_mv', 'circ_mv']:
                        if field in df_basic.columns:
                            result_data[field] = df_basic[field]
                            
            except Exception as e:
                print(f"获取 {ts_code} 估值数据失败: {str(e)}")
                # 继续使用已有的数据

            # 合并所有数据
            if result_data:
                # 确保所有数据长度一致
                min_length = min(len(data) for data in result_data.values())
                if min_length < 100:
                    print(f"⚠️  股票 {ts_code} 有效数据量不足: {min_length} 天")
                    return None
                    
                stock_df = pd.DataFrame(result_data)
                stock_df.columns = [f"{col}_{ts_code}" for col in stock_df.columns]
                print(f"✅ 成功获取 {ts_code} 数据: {len(stock_df)} 天")
                return stock_df
            else:
                return None

        except Exception as e:
            print(f"股票 {ts_code} 获取失败 (第{attempt+1}次重试): {str(e)}")
            time.sleep(3)  # 增加重试间隔
    
    print(f"❌ 股票 {ts_code} 经过 {MAX_RETRIES} 次重试后仍失败")
    return None

def reorganize_data_by_factor(all_stock_data):
    """优化版本的数据组织 - 内存优化"""
    print("🔄 重新组织数据格式...")
    
    if not all_stock_data:
        return {}
    
    # 收集所有日期
    all_dates = set()
    for stock_df in all_stock_data.values():
        if stock_df is not None and len(stock_df) > 0:
            all_dates.update(stock_df.index)
    
    if not all_dates:
        print("❌ 没有有效的日期数据")
        return {}
    
    all_dates = sorted(all_dates)
    print(f"总交易日数: {len(all_dates)}")
    
    # 为每个因子创建空的DataFrame
    base_factors = ['close', 'open', 'high', 'low', 'pct_chg', 'vol', 'amount', 
                   'turnover_rate', 'total_mv', 'circ_mv']
    
    factor_dataframes = {}
    
    # 分批处理，避免内存溢出
    for factor in tqdm(base_factors, desc="组织因子数据"):
        factor_data = pd.DataFrame(index=all_dates)
        
        for ts_code, stock_df in all_stock_data.items():
            if stock_df is not None:
                factor_col = f"{factor}_{ts_code}"
                if factor_col in stock_df.columns:
                    # 直接赋值，避免创建中间对象
                    factor_data[ts_code] = stock_df[factor_col]
        
        # 清理并保存
        factor_data = factor_data.ffill().bfill().fillna(0)
        factor_dataframes[factor] = factor_data
        
        # 强制垃圾回收
        del factor_data
        gc.collect()
    
    return factor_dataframes

def calculate_technical_indicators(factor_dataframes):
    """计算技术指标 - 内存优化版本"""
    print("📊 计算技术指标...")
    
    if 'close' not in factor_dataframes:
        return factor_dataframes
    
    # 分批计算技术指标
    close_data = factor_dataframes['close'].fillna(method='ffill').fillna(method='bfill')
    
    # 计算移动平均线
    for window in [5, 10, 20]:
        ma_name = f'MA{window}'
        factor_dataframes[ma_name] = close_data.rolling(window=window, min_periods=1).mean()
        print(f"✅ 计算 {ma_name}")
    
    # 计算收益率
    factor_dataframes['RETURN_5'] = close_data.pct_change(5, fill_method=None)
    factor_dataframes['RETURN_10'] = close_data.pct_change(10, fill_method=None)
    factor_dataframes['RETURN_20'] = close_data.pct_change(20, fill_method=None)
    
    # 清理内存
    del close_data
    gc.collect()
    
    return factor_dataframes

def calculate_volume_amplification_indicators(factor_dataframes):
    """计算成交量放大指标 - 内存优化版本"""
    print("📈 计算成交量放大指标...")
    
    if 'vol' not in factor_dataframes:
        return factor_dataframes
    
    volume_data = factor_dataframes['vol']
    
    # 分批计算成交量指标
    print("  计算成交量比率...")
    factor_dataframes['VOLUME_RATIO_5'] = volume_data / volume_data.rolling(5, min_periods=1).mean()
    factor_dataframes['VOLUME_RATIO_10'] = volume_data / volume_data.rolling(10, min_periods=1).mean()
    factor_dataframes['VOLUME_RATIO_20'] = volume_data / volume_data.rolling(20, min_periods=1).mean()
    
    print("  计算成交量突破...")
    factor_dataframes['VOLUME_BREAK_5'] = (volume_data > volume_data.rolling(5, min_periods=1).mean() * 1.5).astype(int)
    factor_dataframes['VOLUME_BREAK_10'] = (volume_data > volume_data.rolling(10, min_periods=1).mean() * 1.8).astype(int)
    factor_dataframes['VOLUME_BREAK_20'] = (volume_data > volume_data.rolling(20, min_periods=1).mean() * 2.0).astype(int)
    
    print("  计算成交量动量...")
    factor_dataframes['VOLUME_MOMENTUM_5'] = volume_data / volume_data.shift(5).replace(0, np.nan)
    factor_dataframes['VOLUME_MOMENTUM_10'] = volume_data / volume_data.shift(10).replace(0, np.nan)
    
    # 清理内存
    del volume_data
    gc.collect()
    
    print("✅ 成交量放大指标计算完成")
    return factor_dataframes

def save_data_incremental(factor_dataframes, successful_stocks, chunk_size=50):
    """增量保存数据，避免内存不足 - 修复版本"""
    print("💾 增量保存数据...")
    
    # 创建输出目录
    output_dir = f"stock_data_3year_{len(successful_stocks)}stocks"
    os.makedirs(output_dir, exist_ok=True)
    
    # 在循环前获取日期范围信息
    if factor_dataframes:
        first_factor = list(factor_dataframes.keys())[0]
        first_df = factor_dataframes[first_factor]
        date_range_start = first_df.index.min().strftime('%Y-%m-%d')
        date_range_end = first_df.index.max().strftime('%Y-%m-%d')
    else:
        date_range_start = "未知"
        date_range_end = "未知"
    
    # 分批保存每个因子
    for factor_name, df in tqdm(factor_dataframes.items(), desc="保存因子数据"):
        # 分批保存股票数据
        for i in range(0, len(successful_stocks), chunk_size):
            chunk_stocks = successful_stocks[i:i+chunk_size]
            chunk_df = df[chunk_stocks] if len(chunk_stocks) > 0 else pd.DataFrame()
            
            chunk_file = os.path.join(output_dir, f"{factor_name}_chunk_{i//chunk_size}.pkl")
            with open(chunk_file, 'wb') as f:
                dill.dump(chunk_df, f)
        
        # 清理内存
        del df
        gc.collect()
    
    # 保存元数据
    metadata = {
        'stock_count': len(successful_stocks),
        'date_range': f"{date_range_start} 至 {date_range_end}",
        'factor_count': len(factor_dataframes),
        'creation_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'chunk_size': chunk_size
    }
    
    with open(os.path.join(output_dir, 'metadata.pkl'), 'wb') as f:
        dill.dump(metadata, f)
    
    return output_dir

def clean_and_save_data(factor_dataframes, successful_stocks):
    """数据清洗和保存 - 优化版本"""
    print("🧹 数据清洗中...")
    
    # 统一的缺失值处理
    for factor_name in factor_dataframes:
        df = factor_dataframes[factor_name]
        # 先向前填充，再向后填充，最后用0填充剩余缺失值
        df_cleaned = df.ffill().bfill().fillna(0)
        factor_dataframes[factor_name] = df_cleaned
    
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
    
    # 使用增量保存
    output_dir = save_data_incremental(factor_dataframes, successful_stocks)
    
    # 同时保存完整的收盘价文件（单独文件）
    close_output_file = f'close_price_{len(successful_stocks)}stocks_3year.pkl'
    with open(close_output_file, 'wb') as f:
        dill.dump(factor_dataframes['close'], f)
    
    print(f"\n✅ 数据保存完成!")
    print(f"📁 数据目录: {output_dir}")
    print(f"📁 收盘价文件: {close_output_file}")
    print(f"🎯 特别包含 {len(volume_factors)} 个成交量放大因子")

# ========== 主流程 ==========
if __name__ == '__main__':
    print("开始获取3年选股因子数据...")
    print("=" * 50)
    
    # 获取流动性最好的股票
    valid_stocks = get_valid_stocks_by_liquidity(MAX_STOCKS)
    if not valid_stocks:
        print("❌ 未获取到有效股票列表，程序退出")
        exit()
    
    # 时间范围 - 3年
    end_date = datetime.now().strftime('%Y%m%d')
    start_date = (datetime.now() - timedelta(days=365 * YEARS)).strftime('%Y%m%d')
    print(f"📅 时间范围: {start_date} 至 {end_date} (共{YEARS}年)")
    print(f"📈 股票数量: {len(valid_stocks)}")
    
    # 存储所有股票数据
    all_stock_data = {}
    successful_stocks = []
    
    # 分批获取数据 - 更小的批次和更长的间隔
    total_batches = (len(valid_stocks) + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"\n开始分批获取数据，共 {total_batches} 个批次...")
    
    for i in tqdm(range(0, len(valid_stocks), BATCH_SIZE), desc="总体进度", total=total_batches):
        batch = valid_stocks[i:i + BATCH_SIZE]
        batch_success = 0
        
        with ThreadPoolExecutor(max_workers=3) as executor:  # 减少并发数
            futures = {executor.submit(fetch_single_stock_data, code, start_date, end_date): code for code in batch}
            
            for future in tqdm(futures, desc=f"批次 {i//BATCH_SIZE + 1}", leave=False):
                stock_data = future.result()
                stock_code = futures[future]
                if stock_data is not None:
                    all_stock_data[stock_code] = stock_data
                    successful_stocks.append(stock_code)
                    batch_success += 1
        
        print(f"批次 {i//BATCH_SIZE + 1} 完成: {batch_success}/{len(batch)} 只股票成功")
        
        # 批次间隔避免限速 - 3年数据需要更长的间隔
        if i + BATCH_SIZE < len(valid_stocks):
            wait_time = 15
            print(f"等待{wait_time}秒避免请求限制...")
            time.sleep(wait_time)
            
        # 定期清理内存
        if (i // BATCH_SIZE) % 5 == 0:
            gc.collect()
    
    print(f"\n✅ 数据获取完成，成功获取 {len(successful_stocks)} 只股票数据")
    
    if len(successful_stocks) == 0:
        print("❌ 没有成功获取任何股票数据，程序退出")
        exit()
    
    # 重新组织数据格式
    factor_dataframes = reorganize_data_by_factor(all_stock_data)
    
    if not factor_dataframes:
        print("❌ 没有有效数据，程序退出")
        exit()
    
    # 清理原始数据释放内存
    del all_stock_data
    gc.collect()
    
    # 计算技术指标
    factor_dataframes = calculate_technical_indicators(factor_dataframes)
    
    # 计算成交量放大指标
    factor_dataframes = calculate_volume_amplification_indicators(factor_dataframes)
    
    # 数据清洗和保存
    clean_and_save_data(factor_dataframes, successful_stocks)
    
    print("\n🎉 3年数据获取和处理完成!")