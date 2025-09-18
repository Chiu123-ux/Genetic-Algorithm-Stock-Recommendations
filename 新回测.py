# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager  
import dill
import os
from scipy.stats import spearmanr
from toolkit.setupGPlearn import my_gplearn  # 需确保toolkit模块存在
from tqdm import tqdm  # 添加进度条

# ===================== 中文显示配置（核心修复） =====================
plt.rcParams['font.sans-serif'] = ['SimHei']  
plt.rcParams['axes.unicode_minus'] = False   

# 创建结果目录
if not os.path.exists('./result/factor/'):
    os.makedirs('./result/factor/')

# 1. 因子评价指标（IC值）
def score_func_basic(y, y_pred, sample_weight):
    """Spearman秩相关系数作为评价指标，返回IC绝对值"""
    if len(np.unique(y_pred[-1])) <= 10:  
        return -1
    corr_df = pd.DataFrame(y).corrwith(pd.DataFrame(y_pred), axis=1, method='spearman')
    ic_value = corr_df.mean()
    return abs(ic_value) if not np.isnan(ic_value) else 0

# 2. 更新回测函数（支持动态因子生成和指定开始日期）
def dynamic_factor_backtest(price_df, x_dict, initial_capital=1e6, commission=0.0013, 
                          top_n=5, bottom_n=5, train_window=180, gen_num=3, pop_num=100,
                          start_date=None):
    """
    每月调仓两次（月中和月末）回测，动态训练因子
    :param start_date: 开始调仓的日期 (格式: 'YYYY-MM-DD')
    """
    # 生成调仓日列表（每月15号和月末）
    rebalance_dates = []
    for year_month in pd.date_range(start=price_df.index[0], end=price_df.index[-1], freq='M'):
        # 获取该月的所有交易日
        month_dates = price_df.loc[price_df.index.to_period('M') == year_month.to_period('M')].index
        
        if len(month_dates) == 0:
            continue
        
        # 1. 月末调仓日：该月最后一个交易日
        rebalance_dates.append(month_dates[-1])
        
        # 2. 月中调仓日：找当月15号之前最近的交易日
        mid_date = pd.Timestamp(year=year_month.year, month=year_month.month, day=15)
        if mid_date < price_df.index[0] or mid_date > price_df.index[-1]:
            continue
            
        # 选择15号或之前最近的交易日
        mid_candidates = month_dates[month_dates <= mid_date]
        if len(mid_candidates) > 0:
            rebalance_dates.append(mid_candidates[-1])
    
    # 去重并按时间排序
    rebalance_dates = sorted(set(rebalance_dates))
    
    # 如果指定了开始日期，过滤掉开始日期之前的调仓日
    if start_date:
        start_date = pd.Timestamp(start_date)
        rebalance_dates = [date for date in rebalance_dates if date >= start_date]
        print(f"策略从 {start_date.strftime('%Y-%m-%d')} 开始执行，剩余调仓次数: {len(rebalance_dates)}")
    else:
        print(f"总调仓次数: {len(rebalance_dates)}")
    
    # 账户初始化
    cash = initial_capital
    holdings = {}
    equity_curve = pd.Series(index=price_df.index, dtype=float)
    factor_expressions = {}  # 存储每个调仓日的因子表达式
    all_factors = pd.DataFrame(index=price_df.index, columns=price_df.columns)  # 存储所有因子值
    
    # 函数集配置
    function_set = [
        'add', 'sub', 'mul', 'div', 'sqrt', 'log', 
        'abs', 'neg', 'inv', 'sin', 'cos', 'tan', 
        'max', 'min', 'gtpn', 'andpn', 'orpn', 'ltpn', 
        'gtp', 'andp', 'orp', 'ltp', 'gtn', 'andn', 
        'orn', 'ltn', 'delayy', 'delta', 'signedpower', 
        'decayl', 'stdd', 'rankk'
    ]
    
    # 回测主循环
    for i, date in enumerate(tqdm(price_df.index, desc="回测进度")):
        # 非调仓日只需计算净值
        total_value = cash + sum(
            holdings.get(stock, 0) * price_df.loc[date, stock]
            for stock in holdings
        )
        equity_curve[date] = total_value
        
        # 如果指定了开始日期，且当前日期未到开始日期，跳过调仓
        if start_date and date < start_date:
            continue
            
        if date not in rebalance_dates:
            continue
        
        # 动态因子生成 ==============================================
        # 1. 确定训练窗口（180交易日）
        # 确保训练窗口不会超出数据范围
        idx_loc = price_df.index.get_loc(date)
        if idx_loc < train_window:
            print(f"跳过 {date.strftime('%Y-%m-%d')} 调仓 - 训练窗口不足")
            continue
            
        train_start = price_df.index[idx_loc - train_window]
        train_end = price_df.index[idx_loc - 1]
        
        # 2. 准备训练数据
        price_train = price_df.loc[train_start:train_end]
        
        # 计算训练期的未来20日收益
        ret5_train = np.log(price_train.shift(-20) / price_train)
        y_ret_train = ret5_train.dropna(how='all')
        
        # 对齐特征数据
        x_dict_train = {}
        for key in x_dict:
            # 只取训练窗口内的特征数据
            feat_df = x_dict[key].loc[train_start:train_end].reindex_like(price_train)
            x_dict_train[key] = feat_df.ffill().bfill().fillna(0)  # 三重填充
        
        # 3. 构建训练输入（三维张量）
        feature_names = list(x_dict_train.keys())
        x_array_train = np.array(list(x_dict_train.values()))
        x_array_train = np.transpose(x_array_train, axes=(1, 2, 0))
        
        # 4. 训练因子表达式
        my_cmodel_gp = my_gplearn(
            function_set,
            score_func_basic,
            feature_names=feature_names,
            pop_num=pop_num,
            gen_num=gen_num,
            random_state=3
        )
        my_cmodel_gp.fit(x_array_train, np.array(y_ret_train))
        factor_expression = str(my_cmodel_gp)
        factor_expressions[date] = factor_expression
        
        # 5. 计算当前调仓日的因子值
        current_features = {}
        for key in x_dict:
            current_features[key] = x_dict[key].loc[date].reindex(price_df.columns)
        
        current_x_array = np.array(list(current_features.values()))
        current_x_array = np.transpose(current_x_array, axes=(1, 0))
        current_factor = my_cmodel_gp.predict(current_x_array[np.newaxis, :, :])[0]
        
        # 保存因子值
        factor_series = pd.Series(current_factor, index=price_df.columns)
        all_factors.loc[date] = factor_series
        
        # 修复IC计算错误 ================================
        predicted_factor = my_cmodel_gp.predict(x_array_train)
        ic_values = []
        
        # 逐日计算IC值
        for j, day in enumerate(y_ret_train.index):
            actual_returns = y_ret_train.loc[day].values
            pred_factor = predicted_factor[j, :]
            
            # 移除无效值
            valid_idx = ~np.isnan(actual_returns) & ~np.isnan(pred_factor)
            actual_returns = actual_returns[valid_idx]
            pred_factor = pred_factor[valid_idx]
            
            if len(actual_returns) > 5:  # 确保足够的股票数量
                ic, _ = spearmanr(actual_returns, pred_factor)
                if not np.isnan(ic):
                    ic_values.append(ic)
        
        # 计算平均IC值
        ic_mean = np.mean(ic_values) if ic_values else 0
        # ==============================================
        
        # 调仓逻辑 ==============================================
        # 1. 过滤无效股票（停牌/价格NaN）
        current_price = price_df.loc[date].dropna()
        valid_stocks = current_price.index
        current_factor = factor_series[valid_stocks].dropna()
        
        # 2. 按因子排序选股（Top/Bottom）
        if ic_mean >= 0:
            selected_stocks = current_factor.sort_values(ascending=False).head(top_n).index
            strategy_type = "Top"
        else:
            selected_stocks = current_factor.sort_values(ascending=True).head(bottom_n).index
            strategy_type = "Bottom"
        
        # 3. 打印调仓信息（含因子表达式）
        print(f"\n【调仓日】{date.strftime('%Y-%m-%d')} 策略: {strategy_type}{top_n}")
        print(f"因子表达式: {factor_expression[:100]}...")  # 显示前100字符
        print(f"训练期IC均值: {ic_mean:.4f}")
        print(f"买入股票: {selected_stocks.tolist()}")
        
        # 4. 卖出原有持仓
        for stock in list(holdings.keys()):
            if stock not in price_df.columns:
                continue
                
            sell_price = price_df.loc[date, stock]
            if np.isnan(sell_price) or sell_price <= 0:
                continue
                
            cash += holdings[stock] * sell_price * (1 - commission)
            del holdings[stock]
        
        # 5. 买入新股票（等权重）
        if len(selected_stocks) > 0:
            weight = 1 / len(selected_stocks)
            for stock in selected_stocks:
                buy_price = price_df.loc[date, stock]
                if np.isnan(buy_price) or buy_price <= 0:
                    continue
                    
                qty = (cash * weight) / buy_price
                holdings[stock] = qty
                cash -= qty * buy_price * (1 + commission)
    
    return equity_curve, all_factors, factor_expressions

# 3. 绩效计算函数
def calculate_metrics(equity):
    """计算年化收益、波动率、夏普、最大回撤"""
    returns = equity.pct_change().dropna()
    if len(returns) == 0:
        return {
            '年化收益': 0,
            '年化波动率': 0,
            '夏普比率': 0,
            '最大回撤': 0
        }
    
    annualized_return = (1 + returns.mean()) ** 252 - 1
    annualized_volatility = returns.std() * np.sqrt(252)
    sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility != 0 else np.inf
    max_drawdown = (equity / equity.cummax() - 1).min()
    return {
        '年化收益': annualized_return,
        '年化波动率': annualized_volatility,
        '夏普比率': sharpe_ratio,
        '最大回撤': max_drawdown
    }

# ===================== 主程序 =====================
if __name__ == '__main__':
    # 1. 加载数据
    print("加载数据...")
    with open('stock_close_3years.pkl', 'rb') as f:
        price = dill.load(f)
    price.index = pd.to_datetime(price.index)
    
    with open('factor_3years.pkl', 'rb') as f:
        x_dict = dill.load(f)
    for key in x_dict:
        x_dict[key].index = pd.to_datetime(x_dict[key].index)
    
    # 2. 设置开始日期（例如：'2024-12-15'）
    start_date = '2025-2-15'  # 可以修改为任意日期
    
    # 3. 执行动态因子回测
    print(f"\n开始动态因子回测，从 {start_date} 开始执行...")
    equity_curve, all_factors, factor_expressions = dynamic_factor_backtest(
        price_df=price,
        x_dict=x_dict,
        initial_capital=1e6,
        commission=0.0013,
        top_n=5,
        bottom_n=5,
        train_window=120,
        gen_num=3,  # 可调整以平衡速度与效果
        pop_num=100,  # 可调整以平衡速度与效果
        start_date=start_date  # 传入开始日期
    )
    
    # 4. 保存生成的因子
    all_factors.to_pickle('./result/factor/dynamic_factors.pkl')
    with open('./result/factor/factor_expressions.pkl', 'wb') as f:
        dill.dump(factor_expressions, f)
    print("\n因子数据已保存到./result/factor/目录")
    
    # 5. 绩效分析
    metrics = calculate_metrics(equity_curve)
    print("\n回测绩效指标:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    
    # 6. 净值曲线可视化
    plt.figure(figsize=(12, 6))
    (equity_curve / equity_curve.iloc[0]).plot(title=f'动态因子策略净值曲线（从 {start_date} 开始）')
    plt.xlabel('交易日期')
    plt.ylabel('净值')
    plt.grid(True)
    plt.savefig('./result/factor/equity_curve.png', dpi=300)
    plt.show()