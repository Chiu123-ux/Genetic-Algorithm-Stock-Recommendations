# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager  
import dill
import os
from scipy.stats import spearmanr, zscore
from toolkit.setupGPlearn import my_gplearn  # 需确保toolkit模块存在
from tqdm import tqdm  # 添加进度条

# ===================== 中文显示配置（核心修复） =====================
plt.rcParams['font.sans-serif'] = ['SimHei']  
plt.rcParams['axes.unicode_minus'] = False   

# 创建结果目录
if not os.path.exists('./result/factor/'):
    os.makedirs('./result/factor/')

# 改进后的因子评价指标（支持多因子评估）
def score_func_basic(y, y_pred, sample_weight):
    """Spearman秩相关系数作为评价指标，返回IC绝对值"""
    if len(np.unique(y_pred[-1])) <= 10:  
        return -1
    corr_df = pd.DataFrame(y).corrwith(pd.DataFrame(y_pred), axis=1, method='spearman')
    ic_value = corr_df.mean()
    return abs(ic_value) if not np.isnan(ic_value) else 0

# 改进后的回测函数（支持动态生成多个因子）修复了未来数据泄露
def dynamic_factor_backtest(price_df, x_dict, initial_capital=1e6, commission=0.0013, 
                          top_n=5, bottom_n=5, train_window=180, gen_num=3, pop_num=100,
                          start_date=None, num_factors=3):
    """
    每月调仓两次（月中和月末）回测，动态训练多个因子
    :param start_date: 开始调仓的日期 (格式: 'YYYY-MM-DD')
    :param num_factors: 每次训练生成的因子数量
    """
    # ===================== 修复未来数据泄露 =====================
    # 提前计算整个价格序列的未来10日收益
    future_ret = np.log(price_df.shift(-10) / price_df)
    # =========================================================
    
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
        # 1. 确定训练窗口
        # 确保训练窗口不会超出数据范围
        idx_loc = price_df.index.get_loc(date)
        # ======== 修复：增加偏移量确保未来数据可用 ========
        min_required_window = train_window + 11  # 预留10天用于计算未来收益 + 1天缓冲
        if idx_loc < min_required_window:
            print(f"跳过 {date.strftime('%Y-%m-%d')} 调仓 - 训练窗口不足 (可用:{idx_loc} < 需要:{min_required_window})")
            continue
            
        train_end_idx = idx_loc - 11  # 结束位置提前11天 (预留10天未来收益+1天缓冲)
        train_start_idx = train_end_idx - train_window + 1
        train_start = price_df.index[train_start_idx]
        train_end = price_df.index[train_end_idx]
        # ===============================================
        
        # 2. 准备训练数据
        price_train = price_df.loc[train_start:train_end]
        
        # ======== 修复：使用预计算的未来收益 ========
        # 取训练窗口对应的未来收益
        ret10_train = future_ret.loc[train_start:train_end]
        y_ret_train = ret10_train.dropna(how='all')
        # =========================================
        
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
        
        # 4. 训练多个因子表达式（核心改进）
        factor_models = []
        factor_expr_list = []
        factor_ic_list = []
        
        # 训练多个因子
        for factor_idx in range(num_factors):
            my_cmodel_gp = my_gplearn(
                function_set,
                score_func_basic,
                feature_names=feature_names,
                pop_num=pop_num,
                gen_num=gen_num,
                random_state=3 + factor_idx  # 不同的随机种子生成不同因子
            )
            my_cmodel_gp.fit(x_array_train, np.array(y_ret_train))
            factor_expr = str(my_cmodel_gp)
            
            # 计算因子IC值
            predicted_factor = my_cmodel_gp.predict(x_array_train)
            ic_values = []
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
            factor_ic_list.append(ic_mean)
            factor_models.append(my_cmodel_gp)
            factor_expr_list.append(factor_expr)
            
            # 记录因子表达式（带IC值）
            factor_expressions[f"{date}_factor_{factor_idx+1}"] = {
                "expression": factor_expr,
                "ic": ic_mean
            }
        
        # 5. 计算当前调仓日的因子值并合成
        current_features = {}
        for key in x_dict:
            # 添加三重填充确保数据完整
            current_features[key] = x_dict[key].loc[date].reindex(price_df.columns).ffill().bfill().fillna(0)
        
        # 合成多个因子的综合因子值（加权平均）
        combined_factor = pd.Series(0, index=price_df.columns)
        total_weight = 0
        
        for factor_idx, model in enumerate(factor_models):
            current_x_array = np.array(list(current_features.values()))
            current_x_array = np.transpose(current_x_array, axes=(1, 0))
            current_factor = model.predict(current_x_array[np.newaxis, :, :])[0]
            
            # 标准化因子值
            current_factor = zscore(current_factor, nan_policy='omit')
            current_factor = pd.Series(current_factor, index=price_df.columns)
            
            # 获取因子权重（基于IC绝对值）
            weight = abs(factor_ic_list[factor_idx])
            
            # 调整因子方向：IC为正时因子值越大越好，IC为负时因子值越小越好
            if factor_ic_list[factor_idx] < 0:
                current_factor = -current_factor
            
            # 加权合成
            combined_factor += current_factor * weight
            total_weight += weight
        
        # 避免除以零错误
        if total_weight > 0:
            combined_factor /= total_weight
        else:
            combined_factor = pd.Series(0, index=price_df.columns)
        
        # 保存合成因子值
        all_factors.loc[date] = combined_factor
        
        # 调仓逻辑 ==============================================
        # 1. 过滤无效股票（停牌/价格NaN）
        current_price = price_df.loc[date].dropna()
        valid_stocks = current_price.index
        current_factor = combined_factor[valid_stocks].dropna()
        
        # 2. 按因子排序选股（Top）
        selected_stocks = current_factor.sort_values(ascending=False).head(top_n).index
        strategy_type = "Top"
        
        # 3. 打印调仓信息（含因子表达式）
        print(f"\n【调仓日】{date.strftime('%Y-%m-%d')} 策略: {strategy_type}{top_n}")
        print(f"生成的 {num_factors} 个因子:")
        for idx, expr in enumerate(factor_expr_list):
            print(f"  因子{idx+1} (IC={factor_ic_list[idx]:.4f}): {expr[:80]}...")
        print(f"买入股票: {selected_stocks.tolist()}")
        
        # 4. 卖出原有持仓（只卖出不在新持仓中的股票）
        for stock in list(holdings.keys()):
            if stock not in selected_stocks:
                if stock not in price_df.columns:
                    continue
                    
                sell_price = price_df.loc[date, stock]
                if np.isnan(sell_price) or sell_price <= 0:
                    continue
                    
                cash += holdings[stock] * sell_price * (1 - commission)
                del holdings[stock]
        
        # 5. 买入新股票（等权重）
        # 计算需要买入的股票（只买入当前未持仓的）
        stocks_to_buy = [stock for stock in selected_stocks if stock not in holdings]
        
        if len(stocks_to_buy) > 0:
            weight = 1 / len(stocks_to_buy)
            for stock in stocks_to_buy:
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
    
    # 2. 设置开始日期和因子数量
    start_date = '2024-6-15'  # 可以修改为任意日期
    num_factors = 3  # 每次生成的因子数量
    
    # 3. 执行动态因子回测
    print(f"\n开始动态因子回测，从 {start_date} 开始执行，每次生成 {num_factors} 个因子...")
    equity_curve, all_factors, factor_expressions = dynamic_factor_backtest(
        price_df=price,
        x_dict=x_dict,
        initial_capital=1e6,
        commission=0.0013,
        top_n=5,
        bottom_n=5,
        train_window=30,
        gen_num=3,
        pop_num=100,
        start_date=start_date,
        num_factors=num_factors  # 传入因子数量
    )
    
    
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
    (equity_curve / equity_curve.iloc[0]).plot(title=f'多因子策略净值曲线（从 {start_date} 开始）')
    plt.xlabel('交易日期')
    plt.ylabel('净值')
    plt.grid(True)
    plt.savefig('./result/factor/equity_curve.png', dpi=300)
    plt.show()