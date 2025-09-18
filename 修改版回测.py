# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dill
import os
import re
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
    """每月调仓回测，含数据对齐+选股逻辑+调仓时打印买入股票"""
    # 数据对齐（前向填充）
    factor_df = factor_df.reindex_like(price_df).dropna(how='all')
    price_df.ffill(inplace=True)  
    
    # 获取所有实际存在的年月组合
    available_year_months = factor_df.index.to_period('M').unique()
    rebalance_dates = []
    
    for ym in available_year_months:
        # 获取该年月的所有交易日
        month_df = factor_df.loc[ym.start_time:ym.end_time]
        if not month_df.empty:
            # 取该月最后一个交易日
            last_trading_day = month_df.index[-1]
            rebalance_dates.append(last_trading_day)
    
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
            
            # 打印调仓日与买入股票
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


# 4. 可视化绩效
def visualize_performance(equity_curve, title='因子策略净值曲线'):
    """绘制净值曲线"""
    plt.figure(figsize=(12, 6))
    (equity_curve / equity_curve.iloc[0]).plot(title=title)
    plt.xlabel('交易日期')
    plt.ylabel('净值')
    plt.grid(True)
    plt.show()


# 5. 直接回测已有因子
def run_backtest(factor_df, price_df):
    """直接回测已有因子"""
    # 计算因子IC值
    ret5 = np.log(price_df.shift(-5) / price_df)
    y_ret = ret5.dropna(how='all')
    ic_series = pd.DataFrame(y_ret).corrwith(factor_df, axis=1, method='spearman')
    ic_mean = ic_series.mean()
    print(f"因子 IC 均值：{ic_mean:.4f}")
    
    # 选择Top/Bottom策略
    if ic_mean >= 0:
        print("采用 Top5 选股策略")
        equity_curve = backtest_factor(factor_df, price_df, top_n=5)
    else:
        print("采用 Bottom5 选股策略")
        equity_curve = backtest_factor(factor_df, price_df, bottom_n=5)
    
    # 绩效分析
    metrics = calculate_metrics(equity_curve)
    print("回测绩效指标:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    
    # 可视化
    visualize_performance(equity_curve, title='直接回测-因子策略净值曲线')
    
    return equity_curve, metrics


# 6. 遗传算法生成因子并回测
def run_genetic_algorithm(price_train, price_test, x_dict_train, x_dict_test, function_set, pop_num=500, gen_num=3):
    """运行遗传算法生成因子并进行回测"""
    print("\n" + "="*50)
    print("开始遗传算法因子挖掘")
    print("="*50)
    
    # ---------- 1. 训练集：收益计算+特征构建 ----------
    # 训练集5日收益（当前→未来5天）
    ret5_train = np.log(price_train.shift(-5) / price_train)
    y_ret_train = ret5_train.dropna(how='all')  # 过滤全NaN日期
    
    # 训练集特征对齐（与price_train维度一致）
    feature_names = list(x_dict_train.keys())
    aligned_x_dict_train = {}
    for key in x_dict_train:
        aligned_x_dict_train[key] = x_dict_train[key].reindex_like(price_train).fillna(method='ffill')
    
    # 构建训练集输入张量：(days, stocks, features)
    x_array_train = np.array(list(aligned_x_dict_train.values()))
    x_array_train = np.transpose(x_array_train, axes=(1, 2, 0))  
    
    # ---------- 2. gplearn训练因子（仅训练集） ----------
    my_cmodel_gp = my_gplearn(
        function_set,
        score_func_basic,
        feature_names=feature_names,
        pop_num=pop_num,
        gen_num=gen_num,
        random_state=0
    )
    print("正在运行遗传算法生成因子...")
    my_cmodel_gp.fit(x_array_train, np.array(y_ret_train))
    print(f"生成的因子公式：{my_cmodel_gp}")
    
    # ---------- 3. 测试集：因子预测+数据对齐 ----------
    # 测试集特征对齐（与price_test维度一致）
    aligned_x_dict_test = {}
    for key in x_dict_test:
        aligned_x_dict_test[key] = x_dict_test[key].reindex_like(price_test).fillna(method='ffill')
    
    # 构建测试集输入张量
    x_array_test = np.array(list(aligned_x_dict_test.values()))
    x_array_test = np.transpose(x_array_test, axes=(1, 2, 0))  
    
    # 因子预测（测试集）
    print("正在预测因子值...")
    y_pred_test = my_cmodel_gp.predict(x_array_test)
    factor_df_test = pd.DataFrame(
        y_pred_test,
        index=price_test.index,  
        columns=price_test.columns  
    ).reindex_like(price_test)  # 再次对齐
    
    # ---------- 4. 测试集回测 ----------
    print("\n开始回测生成的因子...")
    equity_curve, metrics = run_backtest(factor_df_test, price_test)
    
    return my_cmodel_gp, factor_df_test, equity_curve, metrics


# 7. 解析因子公式并计算因子值
def parse_and_calculate_factor(factor_str, x_dict, feature_names, dates, stocks):
    """
    解析因子公式并计算因子值
    :param factor_str: 因子公式字符串
    :param x_dict: 原始因子数据字典
    :param feature_names: 特征名称列表
    :param dates: 日期索引
    :param stocks: 股票代码列表
    :return: 计算后的因子DataFrame
    """
    # 初始化因子矩阵
    factor_df = pd.DataFrame(index=dates, columns=stocks)
    
    # 替换公式中的特征名为实际的NumPy数组索引
    # 例如：替换 "f0" 为 "x_array[:, :, 0]"
    for i, feat in enumerate(feature_names):
        factor_str = re.sub(r'\b' + re.escape(feat) + r'\b', f'x_array[:, :, {i}]', factor_str)
    
    # 添加NumPy函数支持
    numpy_funcs = ['np.add', 'np.subtract', 'np.multiply', 'np.divide', 'np.sqrt', 
                  'np.log', 'np.abs', 'np.negative', 'np.reciprocal', 'np.sin', 
                  'np.cos', 'np.tan', 'np.maximum', 'np.minimum']
    
    for func in numpy_funcs:
        alias = func.replace("np.", "")
        factor_str = re.sub(r'\b' + re.escape(alias) + r'\b', func, factor_str)
    
    # 构建计算函数
    try:
        # 创建安全的环境
        env = {
            'np': np,
            'x_array': None  # 将在每个时间点设置
        }
        
        # 编译函数
        code = f"def calculate_factor(x_array):\n    return {factor_str}"
        exec(code, env)
        calculate_factor_func = env['calculate_factor']
        
        # 计算每个时间点的因子值
        for date in dates:
            # 获取该日期的特征数据数组
            date_data = []
            for feat in feature_names:
                # 获取该特征在该日期的值
                feat_values = x_dict[feat].loc[date].values
                date_data.append(feat_values)
            
            # 转换为NumPy数组 (stocks x features)
            x_array_date = np.array(date_data).T  # 转置为(stocks, features)
            
            # 调用函数计算因子值
            try:
                factor_values = calculate_factor_func(x_array_date)
                
                # 确保结果形状正确
                if factor_values.ndim == 0:  # 标量
                    factor_values = np.full(len(stocks), factor_values)
                elif factor_values.ndim == 1 and len(factor_values) == len(stocks):
                    pass  # 形状正确
                else:
                    raise ValueError(f"因子计算结果形状不正确: {factor_values.shape}")
                
                # 存储结果
                factor_df.loc[date] = factor_values
            
            except Exception as e:
                print(f"计算日期 {date} 的因子值时出错: {e}")
                factor_df.loc[date] = np.nan
        
        return factor_df
    
    except Exception as e:
        print(f"解析或计算因子时出错: {e}")
        raise


# ===================== 主程序（交互式） =====================
if __name__ == '__main__':
    # ---------- 加载价格数据 ----------
    print("加载价格数据...")
    with open('stock_close_3years.pkl', 'rb') as f:
        price = dill.load(f)
    price.index = pd.to_datetime(price.index)  # 确保索引可切片
    
    # ---------- 加载因子原始数据 ----------
    print("加载因子原始数据...")
    with open('factor_3years.pkl', 'rb') as f:
        x_dict = dill.load(f)
    for key in x_dict:
        x_dict[key].index = pd.to_datetime(x_dict[key].index)  # 统一为datetime索引
    
    # 按行数分割：前65%训练，后35%测试
    total_days = len(price)
    train_split = int(total_days * 0.65)
    train_dates = price.index[:train_split]
    test_dates = price.index[train_split:]
    
    # 分割训练集与测试集数据
    price_train = price.loc[train_dates]
    price_test = price.loc[test_dates]
    
    x_dict_train = {key: df.loc[train_dates] for key, df in x_dict.items()}
    x_dict_test = {key: df.loc[test_dates] for key, df in x_dict.items()}
    
    # ---------- 函数集配置 ----------
    function_set = [
        'add', 'sub', 'mul', 'div', 'sqrt', 'log', 
        'abs', 'neg', 'inv', 'sin', 'cos', 'tan', 
        'max', 'min',
        # 自定义函数示例（需在my_gplearn中实现）
        # 'gtpn', 'andpn', 'orpn', 'ltpn', 'gtp', 'andp', 'orp', 'ltp', 'gtn', 'andn', 'orn', 'ltn', 'delayy', 'delta', 'signedpower', 'decayl', 'stdd', 'rankk'
    ]
    
    # 获取特征名称列表
    feature_names = list(x_dict.keys())
    
    # ---------- 交互式选择 ----------
    print("\n" + "="*50)
    print("因子回测系统")
    print("="*50)
    
    while True:
        use_existing = input("是否使用已有因子？(yes/no): ").strip().lower()
        
        if use_existing == 'yes':
            # 用户选择使用已有因子
            print("请选择输入方式：")
            print("1. 输入因子数据文件（包含计算好的因子值）")
            print("2. 输入因子公式（文本格式）")
            choice = input("请选择(1/2): ").strip()
            
            if choice == '1':
                # 选项1：直接加载因子值文件
                factor_path = input("因子数据文件路径 (.pkl 文件): ").strip()
                
                if not os.path.exists(factor_path):
                    print(f"错误: 文件 '{factor_path}' 不存在")
                    continue
                
                try:
                    with open(factor_path, 'rb') as f:
                        factor_df = dill.load(f)
                    print(f"成功加载因子数据，维度: {factor_df.shape}")
                except Exception as e:
                    print(f"加载因子数据失败: {e}")
                    continue
                
                # 确保因子数据与价格数据对齐
                factor_df = factor_df.reindex(index=price_test.index, columns=price_test.columns)
                
                # 执行回测
                print("\n" + "="*50)
                print("开始直接回测已有因子")
                print("="*50)
                equity_curve, metrics = run_backtest(factor_df, price_test)
                break
                
            elif choice == '2':
                # 选项2：输入因子公式
                print("\n请输入因子公式（使用特征名称和运算符）：")
                print("可用特征:", ", ".join(feature_names))
                print("示例公式: (close - open) / volume")
                factor_str = input("因子公式: ").strip()
                
                # 解析并计算因子值
                try:
                    print("解析并计算因子值...")
                    factor_df = parse_and_calculate_factor(
                        factor_str, 
                        x_dict_test,  # 使用测试集数据
                        feature_names,
                        test_dates,
                        price_test.columns.tolist()
                    )
                    print(f"因子计算完成，维度: {factor_df.shape}")
                    
                    # 保存因子数据（可选）
                    save_factor = input("是否保存计算的因子数据？(yes/no): ").strip().lower()
                    if save_factor == 'yes':
                        factor_path = input("输入保存路径 (默认: ./result/factor/custom_factor.pkl): ").strip()
                        if not factor_path:
                            factor_path = f"./result/factor/custom_factor_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.pkl"
                        
                        with open(factor_path, 'wb') as f:
                            dill.dump(factor_df, f)
                        print(f"因子数据已保存至: {factor_path}")
                    
                    # 执行回测
                    print("\n" + "="*50)
                    print("开始回测自定义因子")
                    print("="*50)
                    equity_curve, metrics = run_backtest(factor_df, price_test)
                    break
                    
                except Exception as e:
                    print(f"因子计算失败: {e}")
                    continue
                
            else:
                print("无效选择，请输入1或2")
                continue
                
        elif use_existing == 'no':
            # 用户选择运行遗传算法
            print("\n" + "="*50)
            print("开始遗传算法因子挖掘")
            print("="*50)
            
            # 获取遗传算法参数
            pop_num = input(f"输入遗传算法种群规模 (默认500): ").strip()
            gen_num = input(f"输入遗传算法进化代数 (默认3): ").strip()
            
            pop_num = int(pop_num) if pop_num.isdigit() else 500
            gen_num = int(gen_num) if gen_num.isdigit() else 3
            
            # 运行遗传算法
            model, factor_df, equity_curve, metrics = run_genetic_algorithm(
                price_train, price_test, 
                x_dict_train, x_dict_test, 
                function_set,
                pop_num=pop_num,
                gen_num=gen_num
            )
            
            # 保存生成的因子和公式
            save_data = input("是否保存生成的因子和公式？(yes/no): ").strip().lower()
            if save_data == 'yes':
                # 保存因子数据
                factor_path = input("请输入因子数据保存路径 (默认: ./result/factor/generated_factor.pkl): ").strip()
                if not factor_path:
                    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
                    factor_path = f"./result/factor/generated_factor_{timestamp}.pkl"
                
                with open(factor_path, 'wb') as f:
                    dill.dump(factor_df, f)
                print(f"生成的因子数据已保存至: {factor_path}")
                
                # 保存因子公式
                formula_path = input("请输入因子公式保存路径 (默认: ./result/factor/factor_formula.txt): ").strip()
                if not formula_path:
                    formula_path = f"./result/factor/factor_formula_{timestamp}.txt"
                
                with open(formula_path, 'w') as f:
                    f.write(str(model))
                print(f"因子公式已保存至: {formula_path}")
            
            break
            
        else:
            print("输入无效，请输入 'yes' 或 'no'!")
    
    print("\n程序执行完成！")