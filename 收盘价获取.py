import tushare as ts
import pandas as pd
from tqdm import tqdm  # 可选：显示进度条

# 设置 Tushare Token
ts.set_token('')
pro = ts.pro_api()

# ========== 步骤1：获取全A股列表并剔除ST ==========
def get_all_a_stocks():
    df = pro.stock_basic(
        exchange='',
        list_status='L',
        fields='ts_code,name,list_date'
    )
    # 剔除名称含"ST"的股票
    df = df[~df['name'].str.contains('ST')]
    # 排除上市不足5年的股票
    cutoff_date = (pd.to_datetime('today') - pd.DateOffset(years=5)).strftime('%Y%m%d')
    df = df[df['list_date'] < cutoff_date]
    return df['ts_code'].tolist()

valid_stocks = get_all_a_stocks()

# ========== 步骤2：分批获取收盘价数据 ==========
end_date = pd.to_datetime('today').strftime('%Y%m%d')
start_date = (pd.to_datetime(end_date) - pd.DateOffset(years=3)).strftime('%Y%m%d')

# 每批50只股票
batch_size = 50
all_data = []

print(f"开始获取数据（{len(valid_stocks)} 只股票）...")
for i in tqdm(range(0, len(valid_stocks), batch_size), desc="数据下载进度"):
    try:
        batch = valid_stocks[i:i + batch_size]
        ts_code_str = ','.join(batch)
        
        # 仅请求收盘价字段
        df = pro.daily(
            ts_code=ts_code_str,
            start_date=start_date,
            end_date=end_date,
            fields='ts_code,trade_date,close'  # 关键：仅保留收盘价
        )
        all_data.append(df)
    except Exception as e:
        print(f"\n批次 {i//batch_size + 1} 获取失败: {str(e)}")
        continue

# 合并所有批次数据
raw_data = pd.concat(all_data, ignore_index=True)

# ========== 步骤3：数据透视 ==========
# 转换日期格式并去重
raw_data['trade_date'] = pd.to_datetime(raw_data['trade_date'], format='%Y%m%d')
raw_data = raw_data.drop_duplicates(subset=['ts_code', 'trade_date'])

# 透视：日期为索引，股票代码为列，值为收盘价
pivot_data = raw_data.pivot(index='trade_date', columns='ts_code', values='close')

# ========== 步骤4：数据清洗 ==========
# 按日期排序并填充缺失值
pivot_data = pivot_data.sort_index().ffill()

# 剔除全为NaN的列（无效股票）
pivot_data = pivot_data.dropna(axis=1, how='all')

# ========== 步骤5：保存数据 ==========
pivot_data.to_pickle('stock_close_3years.pkl')

print(f"数据已保存，维度：{pivot_data.shape}")
