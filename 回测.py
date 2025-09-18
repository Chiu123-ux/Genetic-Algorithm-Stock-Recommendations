# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager  
import dill
import os
from scipy.stats import spearmanr
from toolkit.setupGPlearn import my_gplearn  # 需确保toolkit模块存在


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


# 2. 回测函数（支持 Top/Bottom 选股 + 调仓时打印买入股票）
def backtest_factor(factor_df, price_df, top_n=10, bottom_n=0, initial_capital=1e6, commission=0.0013):
    """每月调仓两次（月中和月末）回测，含数据对齐+选股逻辑+调仓时打印买入股票"""
    # 数据对齐（前向填充）
    factor_df = factor_df.reindex_like(price_df).dropna(how='all')
    price_df.ffill(inplace=True)  
    
    # ===== 修改：每月两次调仓（月中15号附近和月末） =====
    rebalance_dates = []
    
    # 按月分组
    for year_month, group in factor_df.groupby(pd.Grouper(freq='M')):
        if len(group) == 0:
            continue
        
        # 获取该月的所有交易日
        trading_days = group.index
        
        # 1. 月末调仓日：该月最后一个交易日
        end_of_month = trading_days[-1]
        rebalance_dates.append(end_of_month)
        
        # 2. 月中调仓日：找当月15号之前最近的交易日
        mid_month = pd.Timestamp(year_month.year, year_month.month, 15)
        
        # 如果15号是交易日，直接使用
        if mid_month in trading_days:
            rebalance_dates.append(mid_month)
        else:
            # 找15号之前最近的交易日（优先选择之前的交易日）
            before_15 = trading_days[trading_days < mid_month]
            if len(before_15) > 0:
                # 选择15号之前最近的交易日（即最后一个小于15号的交易日）
                rebalance_dates.append(before_15[-1])
    
    # 去重并按时间排序
    rebalance_dates = sorted(set(rebalance_dates))
    
    # 账户初始化
    cash = initial_capital
    holdings = {}  
    equity_curve = pd.Series(index=price_df.index, dtype=float)
    
    for date in price_df.index:
        if date in rebalance_dates:
            # 1. 过滤无效股票（停牌/价格NaN）
            current_factor = factor_df.loc[date].dropna()
            current_price = price_df.loc[date][current_factor.index].dropna()
            valid_stocks = current_price.index
            current_factor = current_factor[valid_stocks]
            
            # 2. 按因子排序选股（Top/Bottom）
            if top_n > 0:
                selected_stocks = current_factor.sort_values(ascending=False).head(top_n).index
            elif bottom_n > 0:
                selected_stocks = current_factor.sort_values(ascending=True).head(bottom_n).index
            else:
                raise ValueError("top_n / bottom_n 必须>0")
            
            # ---------- 打印调仓日与买入股票 ----------
            print(f"【调仓日】{date.strftime('%Y-%m-%d')} 买入股票：{selected_stocks.tolist()}")
            
            # 3. 卖出原有持仓
            for stock in list(holdings.keys()):
                sell_price = price_df.loc[date, stock]
                if np.isnan(sell_price) or sell_price <= 0:
                    continue
                cash += holdings[stock] * sell_price * (1 - commission)
                del holdings[stock]
            
            # 4. 买入新股票（等权重）
            if len(selected_stocks) > 0:
                weight = 1 / len(selected_stocks)
                for stock in selected_stocks:
                    buy_price = price_df.loc[date, stock]
                    if np.isnan(buy_price) or buy_price <= 0:
                        continue
                    qty = (cash * weight) / buy_price
                    holdings[stock] = qty
                    cash -= qty * buy_price * (1 + commission)
        
        # 计算当前净值
        total_value = cash + sum(
            holdings.get(stock, 0) * price_df.loc[date, stock]
            for stock in holdings
        )
        equity_curve[date] = total_value
    
    return equity_curve


# 3. 绩效计算函数
def calculate_metrics(equity):
    """计算年化收益、波动率、夏普、最大回撤"""
    returns = equity.pct_change().dropna()
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


# ===================== 主程序：行数分割训练+回测 =====================
if __name__ == '__main__':
    # ---------- 1. 函数集配置 ----------
    function_set = [
        'add', 'sub', 'mul', 'div', 'sqrt', 'log', 
        'abs', 'neg', 'inv', 'sin', 'cos', 'tan', 
        'max', 'min',
        # 若需更多自定义函数，取消注释下方行并补充
        'gtpn', 'andpn', 'orpn', 'ltpn', 'gtp', 'andp', 'orp', 'ltp', 'gtn', 'andn', 'orn', 'ltn', 'delayy', 'delta', 'signedpower', 'decayl', 'stdd', 'rankk'
    ]
    
    # ---------- 2. 数据加载与行数分割 ----------
    # 加载价格数据（强制转换为datetime索引）
    with open('stock_close_3years.pkl', 'rb') as f:
        price = dill.load(f)
    price.index = pd.to_datetime(price.index)  # 确保索引可切片
    
    # 加载因子原始数据（open/close/vol等，同样处理索引）
    with open('factor_3years.pkl', 'rb') as f:
        x_dict = dill.load(f)
    for key in x_dict:
        x_dict[key].index = pd.to_datetime(x_dict[key].index)  # 统一为datetime索引
    
    # 按行数分割：前85%训练，后15%测试（可根据需求调整分割比例）
    total_days = len(price)
    train_split = int(total_days * 0.65)
    train_dates = price.index[:train_split]
    test_dates = price.index[train_split:]
    
    # 分割训练集与测试集数据
    price_train = price.loc[train_dates]
    price_test = price.loc[test_dates]
    
    x_dict_train = {key: df.loc[train_dates] for key, df in x_dict.items()}
    x_dict_test = {key: df.loc[test_dates] for key, df in x_dict.items()}
    
    # ---------- 3. 训练集：收益计算+特征构建 ----------
    # 训练集10日收益（当前→未来10天）
    ret5_train = np.log(price_train.shift(-10) / price_train)
    y_ret_train = ret5_train.dropna(how='all')  # 过滤全NaN日期
    
    # 训练集特征对齐（与price_train维度一致）
    feature_names = list(x_dict_train.keys())
    aligned_x_dict_train = {}
    for key in x_dict_train:
        aligned_x_dict_train[key] = x_dict_train[key].reindex_like(price_train).fillna(method='ffill')
    
    # 构建训练集输入张量：(days, stocks, features)
    x_array_train = np.array(list(aligned_x_dict_train.values()))
    x_array_train = np.transpose(x_array_train, axes=(1, 2, 0))  
    
    # ---------- 4. gplearn训练因子（仅训练集） ----------
    my_cmodel_gp = my_gplearn(
        function_set,
        score_func_basic,
        feature_names=feature_names,
        pop_num=100,   # 种群规模
        gen_num=3,    # 进化代数
        random_state=3
    )
    my_cmodel_gp.fit(x_array_train, np.array(y_ret_train))
    print(f"生成的因子公式：{my_cmodel_gp}")
    
    # ---------- 5. 测试集：因子预测+数据对齐 ----------
    # 测试集特征对齐（与price_test维度一致）
    aligned_x_dict_test = {}
    for key in x_dict_test:
        aligned_x_dict_test[key] = x_dict_test[key].reindex_like(price_test).fillna(method='ffill')
    
    # 构建测试集输入张量
    x_array_test = np.array(list(aligned_x_dict_test.values()))
    x_array_test = np.transpose(x_array_test, axes=(1, 2, 0))  
    
    # 因子预测（测试集）
    y_pred_test = my_cmodel_gp.predict(x_array_test)
    factor_df_test = pd.DataFrame(
        y_pred_test,
        index=price_test.index,  
        columns=price_test.columns  
    ).reindex_like(price_test)  # 再次对齐
    
    # ---------- 6. 测试集：IC判断+回测 ----------
    # 测试集10日收益（用于IC计算）
    ret5_test = np.log(price_test.shift(-10) / price_test)
    y_ret_test = ret5_test.dropna(how='all')
    
    # 计算测试集因子IC均值
    ic_series_test = pd.DataFrame(y_ret_test).corrwith(factor_df_test, axis=1, method='spearman')
    ic_mean_test = ic_series_test.mean()
    print(f"测试集因子 IC 均值：{ic_mean_test:.4f}")
    
    # 选择Top/Bottom策略
    if ic_mean_test >= 0:
        print("采用 Top5 选股策略")
        equity_curve = backtest_factor(factor_df_test, price_test, top_n=5)
    else:
        print("采用 Bottom5 选股策略")
        equity_curve = backtest_factor(factor_df_test, price_test, bottom_n=5)
    
    # ---------- 7. 绩效分析+可视化（中文正常显示） ----------
    metrics = calculate_metrics(equity_curve)
    print("回测绩效指标:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    
    # 净值曲线可视化（标题、标签已支持中文）
    plt.figure(figsize=(12, 6))
    (equity_curve / equity_curve.iloc[0]).plot(title='因子策略净值曲线（每月两次调仓）')
    plt.xlabel('交易日期')
    plt.ylabel('净值')
    plt.grid(True)
    plt.show()