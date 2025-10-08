# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from scipy.stats import zscore
import dill
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib as mpl

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

class SmartVolumeBacktester:
    """
    智能成交量增强策略回测系统 - 20日调仓周期
    """
    
    def __init__(self, initial_capital=1000000, transaction_cost=0.001):
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        
        # 回测记录
        self.portfolio_values = []
        self.dates = []
        self.positions = {}  # 当前持仓 {股票代码: 股数}
        self.trade_history = []  # 交易记录
        self.holdings_history = []  # 每日持仓记录
        self.cash_history = []  # 每日现金记录
        
        # 绩效指标
        self.performance_metrics = {}
        
    def run_backtest(self, price_df, x_dict, rebalance_days=20, top_n=10, lookback_months=6):
        """
        运行回测 - 20日调仓周期
        
        Parameters:
        -----------
        price_df: DataFrame
            价格数据
        x_dict: dict
            因子数据字典
        rebalance_days: int
            调仓周期（天数）- 改为20天
        top_n: int
            每次选股数量
        lookback_months: int
            训练数据回溯月数
        """
        print("🎯 开始回测 (20日调仓周期)...")
        
        # 生成调仓日期 - 每20天调仓
        all_dates = price_df.index
        rebalance_dates = all_dates[::rebalance_days]
        
        # 确保有足够的数据开始回测
        start_idx = max(1, len(all_dates) // 10)  # 从10%位置开始，确保有足够训练数据
        rebalance_dates = [date for date in rebalance_dates if date >= all_dates[start_idx]]
        
        print(f"回测期间: {rebalance_dates[0].strftime('%Y-%m-%d')} 至 {rebalance_dates[-1].strftime('%Y-%m-%d')}")
        print(f"调仓次数: {len(rebalance_dates)}")
        print(f"调仓周期: {rebalance_days}天")
        
        current_cash = self.initial_capital
        predictor = SmartVolumeEnhancedPredictor()
        
        # 添加进度条
        for i, date in enumerate(tqdm(rebalance_dates, desc="回测进度")):
            current_date_idx = all_dates.get_loc(date)
            
            # 🔧 关键修复：确保训练数据严格在调仓日期之前
            # 训练数据截止到调仓日期的前一天
            train_end_idx = current_date_idx - 1  # 使用前一天的数据
            if train_end_idx < 60:  # 确保有足够的数据
                continue
                
            train_start_idx = max(0, train_end_idx - 30 * lookback_months)  # 回溯6个月
            
            train_dates = all_dates[train_start_idx:train_end_idx]
            
            if len(train_dates) < 60:  # 至少需要60天数据
                continue
                
            # 🔧 关键修复：训练数据严格使用历史数据
            price_train = price_df.iloc[train_start_idx:train_end_idx]
            
            # 准备因子数据
            x_dict_train = {}
            for factor_name, factor_data in x_dict.items():
                factor_train = factor_data.iloc[train_start_idx:train_end_idx]
                # 重新索引以确保对齐
                factor_train = factor_train.reindex(price_train.index)
                factor_train = factor_train.ffill().bfill().fillna(0)
                x_dict_train[factor_name] = factor_train
            
            # 2. 训练模型 - 预测20天收益率
            success = predictor.robust_quick_train(
                price_df=price_train,
                x_dict=x_dict_train,
                train_months=lookback_months,
                predict_days=rebalance_days
            )
            
            # 3. 获取推荐股票
            if success:
                # 🔧 关键修复：使用训练期间的最后一天数据进行预测（严格历史数据）
                price_for_prediction = price_train
                x_dict_for_prediction = x_dict_train
                
                recommendations = predictor.predict_with_smart_volume_enhancement(
                    price_df=price_for_prediction,
                    x_dict=x_dict_for_prediction,
                    top_n=top_n
                )
            else:
                # 使用备选策略
                recommendations = predictor.volume_based_fallback(
                    price_train, x_dict_train, top_n
                )
            
            # 4. 执行调仓
            if recommendations is not None and len(recommendations) > 0:
                current_cash = self.rebalance_portfolio(
                    date, recommendations, price_df, current_cash
                )
            
            # 5. 记录每日组合价值
            self.record_daily_portfolio(date, price_df, current_cash)
        
        # 6. 计算绩效指标
        self.calculate_performance_metrics(price_df)
        
        print("✅ 回测完成!")
        return self.performance_metrics    
        
    def rebalance_portfolio(self, date, recommendations, price_df, current_cash):
        """执行调仓操作"""
        # 获取当日价格
        if date not in price_df.index:
            return current_cash
            
        current_prices = price_df.loc[date]
        
        # 1. 清空现有持仓
        total_sell_value = 0
        stocks_to_sell = list(self.positions.keys())
        
        for stock in stocks_to_sell:
            if stock in current_prices and not np.isnan(current_prices[stock]):
                price = current_prices[stock]
                shares = self.positions[stock]
                sell_value = shares * price * (1 - self.transaction_cost)  # 扣除卖出手续费
                total_sell_value += sell_value
                
                # 记录卖出交易
                self.trade_history.append({
                    'date': date,
                    'stock': stock,
                    'action': 'SELL',
                    'shares': shares,
                    'price': price,
                    'value': sell_value,
                    'type': 'rebalance'
                })
        
        # 清空持仓
        self.positions = {}
        current_cash += total_sell_value
        
        # 2. 买入新推荐的股票
        if len(recommendations) == 0:
            return current_cash
            
        # 等权重分配资金
        capital_per_stock = current_cash / len(recommendations)
        
        for stock, score in recommendations.items():
            if stock in current_prices and not np.isnan(current_prices[stock]):
                price = current_prices[stock]
                if price > 0:
                    # 计算可买股数（扣除买入手续费）
                    max_shares = int(capital_per_stock / (price * (1 + self.transaction_cost)))
                    if max_shares > 0:
                        cost = max_shares * price * (1 + self.transaction_cost)
                        if cost <= current_cash:
                            self.positions[stock] = max_shares
                            current_cash -= cost
                            
                            # 记录买入交易
                            self.trade_history.append({
                                'date': date,
                                'stock': stock,
                                'action': 'BUY',
                                'shares': max_shares,
                                'price': price,
                                'value': max_shares * price,
                                'type': 'rebalance'
                            })
        
        return current_cash
    
    def record_daily_portfolio(self, date, price_df, current_cash):
        """记录每日组合价值"""
        if date not in price_df.index:
            return
            
        current_prices = price_df.loc[date]
        portfolio_value = current_cash
        
        # 计算持仓价值
        for stock, shares in self.positions.items():
            if stock in current_prices and not np.isnan(current_prices[stock]):
                portfolio_value += shares * current_prices[stock]
        
        self.portfolio_values.append(portfolio_value)
        self.dates.append(date)
        self.cash_history.append(current_cash)
        
        # 记录当前持仓
        holdings = {stock: {'shares': shares, 'price': current_prices.get(stock, 0)} 
                   for stock, shares in self.positions.items()}
        self.holdings_history.append({
            'date': date,
            'holdings': holdings,
            'cash': current_cash,
            'total_value': portfolio_value
        })
    
    def calculate_performance_metrics(self, price_df):
        """计算回测绩效指标"""
        if len(self.portfolio_values) < 2:
            print("❌ 回测数据不足，无法计算绩效")
            return {}
        
        # 转换为Series方便计算
        portfolio_series = pd.Series(self.portfolio_values, index=self.dates)
        
        # 计算收益率
        daily_returns = portfolio_series.pct_change().dropna()
        
        # 基本指标
        total_return = (portfolio_series.iloc[-1] / self.initial_capital - 1) * 100
        total_days = (portfolio_series.index[-1] - portfolio_series.index[0]).days
        annual_return = (1 + total_return/100) ** (365 / total_days) - 1 if total_days > 0 else 0
        
        # 风险指标
        volatility = daily_returns.std() * np.sqrt(252) * 100  # 年化波动率
        sharpe_ratio = annual_return / (volatility/100) if volatility > 0 else 0
        
        # 最大回撤
        max_drawdown = self.calculate_max_drawdown(portfolio_series)
        
        # 交易统计
        trade_df = pd.DataFrame(self.trade_history)
        if len(trade_df) > 0:
            buy_trades = trade_df[trade_df['action'] == 'BUY']
            sell_trades = trade_df[trade_df['action'] == 'SELL']
            total_trades = len(trade_df)
            
            # 计算胜率（需要更复杂的逻辑，这里简化）
            winning_trades = 0
            for i, sell_trade in sell_trades.iterrows():
                # 找到对应的买入交易
                buy_trade = buy_trades[
                    (buy_trades['stock'] == sell_trade['stock']) & 
                    (buy_trades['date'] < sell_trade['date'])
                ].iloc[-1] if not buy_trades.empty else None
                
                if buy_trade is not None and sell_trade['price'] > buy_trade['price']:
                    winning_trades += 1
            
            win_rate = (winning_trades / len(sell_trades)) * 100 if len(sell_trades) > 0 else 0
        else:
            total_trades = 0
            win_rate = 0
        
        # 保存绩效指标
        self.performance_metrics = {
            '初始资金': self.initial_capital,
            '最终净值': portfolio_series.iloc[-1],
            '总收益率': f"{total_return:.2f}%",
            '年化收益率': f"{annual_return*100:.2f}%",
            '年化波动率': f"{volatility:.2f}%",
            '夏普比率': f"{sharpe_ratio:.2f}",
            '最大回撤': f"{max_drawdown:.2f}%",
            '总交易次数': total_trades,
            '胜率': f"{win_rate:.1f}%",
            '回测期间': f"{portfolio_series.index[0].strftime('%Y-%m-%d')} 至 {portfolio_series.index[-1].strftime('%Y-%m-%d')}",
            '回测天数': len(portfolio_series),
            '调仓周期': '20个交易日'
        }
        
        return self.performance_metrics
    
    def calculate_max_drawdown(self, portfolio_series):
        """计算最大回撤"""
        rolling_max = portfolio_series.expanding().max()
        drawdown = (portfolio_series - rolling_max) / rolling_max
        max_drawdown = drawdown.min() * 100
        return max_drawdown
    
    def plot_backtest_results(self, benchmark_data=None, benchmark_name="基准"):
        """绘制回测结果图表"""
        if len(self.portfolio_values) == 0:
            print("❌ 没有回测数据可绘制")
            return
        
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 净值曲线
        portfolio_series = pd.Series(self.portfolio_values, index=self.dates)
        normalized_portfolio = portfolio_series / self.initial_capital
        
        axes[0, 0].plot(normalized_portfolio.index, normalized_portfolio.values, 
                       linewidth=2, label='策略净值', color='#2E86AB')
        
        # 如果有基准数据，添加基准
        if benchmark_data is not None:
            benchmark_series = benchmark_data.reindex(self.dates).ffill().bfill()
            if len(benchmark_series) > 0:
                normalized_benchmark = benchmark_series / benchmark_series.iloc[0]
                axes[0, 0].plot(normalized_benchmark.index, normalized_benchmark.values,
                               linewidth=2, label=benchmark_name, color='#A23B72', linestyle='--')
        
        axes[0, 0].set_title('净值曲线 (20日调仓)', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylabel('累计收益')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 回撤曲线
        rolling_max = portfolio_series.expanding().max()
        drawdown = (portfolio_series - rolling_max) / rolling_max * 100
        
        axes[0, 1].fill_between(drawdown.index, drawdown.values, 0, 
                               color='#F18F01', alpha=0.7)
        axes[0, 1].set_title('回撤曲线', fontsize=14, fontweight='bold')
        axes[0, 1].set_ylabel('回撤 (%)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 月度收益率热力图
        monthly_returns = self.calculate_monthly_returns(portfolio_series)
        if monthly_returns is not None:
            sns.heatmap(monthly_returns, annot=True, fmt='.1%', cmap='RdYlGn', 
                       center=0, ax=axes[1, 0])
            axes[1, 0].set_title('月度收益率 (%)', fontsize=14, fontweight='bold')
        
        # 4. 调仓周期收益率分布
        rebalance_returns = self.calculate_rebalance_returns(portfolio_series)
        if rebalance_returns is not None:
            axes[1, 1].hist(rebalance_returns, bins=20, color='#C73E1D', alpha=0.7, edgecolor='black')
            axes[1, 1].axvline(rebalance_returns.mean(), color='red', linestyle='--', 
                              label=f'均值: {rebalance_returns.mean():.2f}%')
            axes[1, 1].set_title('20日调仓周期收益率分布', fontsize=14, fontweight='bold')
            axes[1, 1].set_xlabel('20日收益率 (%)')
            axes[1, 1].set_ylabel('频数')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def calculate_monthly_returns(self, portfolio_series):
        """计算月度收益率"""
        try:
            monthly_returns = portfolio_series.resample('M').last().pct_change()
            monthly_returns = monthly_returns.dropna()
            
            # 创建月度收益率表格
            monthly_table = pd.DataFrame({
                'Year': monthly_returns.index.year,
                'Month': monthly_returns.index.month,
                'Return': monthly_returns
            })
            
            monthly_pivot = monthly_table.pivot(index='Year', columns='Month', values='Return')
            monthly_pivot.columns = ['1月', '2月', '3月', '4月', '5月', '6月', 
                                   '7月', '8月', '9月', '10月', '11月', '12月']
            
            return monthly_pivot
        except:
            return None
    
    def calculate_rebalance_returns(self, portfolio_series):
        """计算调仓周期收益率"""
        try:
            # 每20个交易日取一个点
            rebalance_points = portfolio_series.iloc[::20]
            rebalance_returns = rebalance_points.pct_change().dropna() * 100
            return rebalance_returns
        except:
            return None
    
    def generate_report(self):
        """生成回测报告"""
        if not self.performance_metrics:
            print("❌ 请先运行回测")
            return
        
        print("\n" + "="*70)
        print("📊 智能成交量增强策略回测报告 (20日调仓)")
        print("="*70)
        
        for key, value in self.performance_metrics.items():
            print(f"{key:>15}: {value}")
        
        print("\n📈 最近10次交易记录:")
        if self.trade_history:
            recent_trades = pd.DataFrame(self.trade_history[-10:])
            print(recent_trades.to_string(index=False))
        else:
            print("无交易记录")
        
        print("="*70)
    
    def save_backtest_results(self, filename=None):
        """保存回测结果"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f'smart_volume_20day_backtest_results_{timestamp}.pkl'
        
        results = {
            'portfolio_values': self.portfolio_values,
            'dates': self.dates,
            'trade_history': self.trade_history,
            'holdings_history': self.holdings_history,
            'performance_metrics': self.performance_metrics,
            'parameters': {
                'initial_capital': self.initial_capital,
                'transaction_cost': self.transaction_cost,
                'rebalance_days': 20
            }
        }
        
        with open(filename, 'wb') as f:
            dill.dump(results, f)
        
        print(f"💾 回测结果已保存到: {filename}")

class SmartVolumeEnhancedPredictor:
    """
    智能成交量增强的股票预测器 - 20日预测版本
    """
    
    def __init__(self):
        self.function_set = ['add', 'sub', 'mul', 'div', 'sqrt', 'log', 'abs', 'neg', 'inv', 'max', 'min', 'rankk']
        self.model = None
        self.factor_expression = None
        self.final_raw_ic = 0
        self.final_abs_ic = 0
        self.min_acceptable_ic = 0.01
        self.direction_consistency = 0
        self.ic_std = 0
        self.dominant_direction = "正向"
        
    def load_and_prepare_data(self, price_file, factor_file):
        """加载价格和因子数据 - 兼容新旧格式"""
        print("📂 加载历史数据...")
        
        try:
            # 1. 加载价格数据
            with open(price_file, 'rb') as f:
                price_data = dill.load(f)
            
            # 处理价格数据格式
            if isinstance(price_data, pd.DataFrame):
                price_df = price_data
                print("✅ 价格数据格式: 旧格式(DataFrame)")
            elif isinstance(price_data, dict) and 'factors' in price_data:
                if 'close' in price_data['factors']:
                    price_df = price_data['factors']['close']
                    print("✅ 价格数据格式: 新格式(字典->factors->close)")
                else:
                    raise ValueError("价格数据中未找到close因子")
            elif isinstance(price_data, dict) and 'close' in price_data:
                price_df = price_data['close']
                print("✅ 价格数据格式: 兼容格式(字典->close)")
            else:
                raise ValueError(f"未知的价格数据格式: {type(price_data)}")
            
            price_df.index = pd.to_datetime(price_df.index)
            
            # 2. 加载因子数据
            with open(factor_file, 'rb') as f:
                factor_data = dill.load(f)
            
            # 处理因子数据格式
            x_dict = {}
            if isinstance(factor_data, dict) and 'factors' in factor_data:
                x_dict = factor_data['factors']
                print("✅ 因子数据格式: 新格式(字典->factors)")
            elif isinstance(factor_data, dict):
                if all(isinstance(v, pd.DataFrame) for v in factor_data.values()):
                    x_dict = factor_data
                    print("✅ 因子数据格式: 旧格式(因子字典)")
                else:
                    raise ValueError("无法识别因子数据格式")
            else:
                raise ValueError(f"未知的因子数据格式: {type(factor_data)}")
            
            # 确保所有因子数据的索引是datetime类型并与价格数据对齐
            for key in x_dict:
                x_dict[key].index = pd.to_datetime(x_dict[key].index)
                x_dict[key] = x_dict[key].reindex(price_df.index)
            
            print(f"✅ 数据加载完成")
            print(f"   时间范围: {price_df.index[0].strftime('%Y-%m-%d')} 至 {price_df.index[-1].strftime('%Y-%m-%d')}")
            print(f"   股票数量: {len(price_df.columns)}")
            print(f"   因子数量: {len(x_dict)}")
            
            # 特别显示成交量因子
            volume_factors = [f for f in x_dict.keys() if 'VOLUME' in f.upper() or 'VOL' in f.upper()]
            print(f"   📈 成交量因子: {len(volume_factors)}个")
            
            return price_df, x_dict
            
        except Exception as e:
            print(f"❌ 数据加载失败: {e}")
            raise
    
    def robust_quick_train(self, price_df, x_dict, train_months=6, predict_days=20):
        """修复版的快速训练 - 严格避免未来数据"""
        print(f"\n🎯 开始快速训练 (预测{predict_days}天)...")
        print(f"   训练周期: {train_months}个月")
        print(f"   预测目标: 未来{predict_days}天表现")
        
        # 确定训练窗口 - 使用严格的历史数据
        end_date = price_df.index[-1]
        start_date = price_df.index[0]
        
        # 准备训练数据 - 确保数据对齐
        price_train = price_df.loc[start_date:end_date]
        
        print(f"   价格数据形状: {price_train.shape}")
        
        # 计算未来收益率 - 严格避免未来数据泄露
        print("📊 计算未来收益率目标...")
        future_returns = []
        valid_dates = []
        
        # 确保我们有足够的数据点
        if len(price_train) <= predict_days:
            print(f"❌ 训练数据不足: 需要至少{predict_days+1}天数据，当前只有{len(price_train)}天")
            return False
        
        for i in range(len(price_train) - predict_days):
            current_date = price_train.index[i]
            future_date = price_train.index[i + predict_days]
            
            daily_returns = {}
            valid_stock_count = 0
            
            for stock in price_train.columns:
                current_price = price_train.iloc[i][stock]
                future_price = price_train.iloc[i + predict_days][stock]
                
                # 严格的数据验证
                if (not np.isnan(current_price) and not np.isnan(future_price) and 
                    current_price > 0 and future_price > 0):
                    future_ret = (future_price - current_price) / current_price
                    daily_returns[stock] = future_ret
                    valid_stock_count += 1
            
            # 只有当有足够多有效股票时才保留该日期
            if valid_stock_count > len(price_train.columns) * 0.8:  # 至少80%的股票有效
                future_returns.append(daily_returns)
                valid_dates.append(current_date)
        
        if not future_returns:
            print("❌ 没有足够有效的未来收益率数据")
            return False
        
        # 创建目标变量 - 确保维度一致
        y_target = pd.DataFrame(future_returns, index=valid_dates)
        y_target = y_target.reindex(columns=price_train.columns)  # 确保列顺序一致
        y_target = y_target.fillna(0)  # 填充缺失值
        
        print(f"✅ 收益率目标计算完成，有效天数: {len(y_target)}")
        print(f"   目标数据形状: {y_target.shape}")
        
        # 🔧 关键修复：特征数据必须与目标数据严格对齐
        # 只使用目标数据对应的日期
        feature_dates = y_target.index
        
        # 准备特征数据 - 选择最重要的因子
        print("🔧 准备特征数据...")
        
        # 优先选择成交量相关因子
        volume_related_factors = [f for f in x_dict.keys() if 'VOLUME' in f.upper() or 'VOL' in f.upper()]
        # 选择其他重要因子，但控制总数
        other_important_factors = ['close', 'pct_chg', 'turnover_rate', 'total_mv', 'high', 'low', 'open']
        other_factors = [f for f in other_important_factors if f in x_dict]
        
        # 组合因子，总数控制在12个以内
        selected_factors = volume_related_factors[:8] + other_factors[:4]
        selected_factors = selected_factors[:12]  # 确保不超过12个
        
        print(f"   使用因子: {len(selected_factors)}个")
        
        x_dict_train = {}
        for key in selected_factors:
            # 🔧 关键修复：特征数据只使用目标数据对应的日期
            feat_df = x_dict[key].loc[feature_dates]
            # 严格的数据清洗
            feat_df = feat_df.ffill().bfill().fillna(0)
            feat_df = feat_df.replace([np.inf, -np.inf], 0)
            x_dict_train[key] = feat_df
        
        # 构建训练数据 - 确保维度一致
        x_array_list = []
        for key in selected_factors:
            # 只取共同日期的数据
            feat_data = x_dict_train[key]
            x_array_list.append(feat_data.values)
        
        # 检查所有特征数据的形状是否一致
        shapes = [arr.shape for arr in x_array_list]
        if len(set(shapes)) != 1:
            print(f"❌ 特征数据形状不一致: {shapes}")
            return False
        
        x_array_train = np.array(x_array_list)
        x_array_train = np.transpose(x_array_train, axes=(1, 2, 0))  # (时间, 股票, 因子)
        
        # 目标数据
        y_target_valid = y_target.values
        
        print(f"✅ 训练数据准备完成")
        print(f"   特征数据形状: {x_array_train.shape}")
        print(f"   目标数据形状: {y_target_valid.shape}")
        
        # 简化的评分函数 - 更稳定
        def stable_score_func(y, y_pred, sample_weight):
            try:
                if len(np.unique(y_pred[-1])) <= 5:
                    return -1
                
                all_ics = []
                for day_idx in range(min(len(y), len(y_pred))):  # 确保不越界
                    if day_idx >= len(y_pred):
                        break
                        
                    df_day = pd.DataFrame({
                        'true': y[day_idx], 
                        'pred': y_pred[day_idx]
                    })
                    # 严格的数据清洗
                    df_day = df_day.replace([np.inf, -np.inf], np.nan)
                    df_day = df_day.dropna()
                    
                    if len(df_day) > 10:
                        ic_day = df_day['true'].corr(df_day['pred'], method='spearman')
                        if not np.isnan(ic_day):
                            all_ics.append(abs(ic_day))
                
                if not all_ics:
                    return 0
                
                return np.mean(all_ics)
            except Exception as e:
                return 0
        
        # 训练模型 - 使用更保守的参数
        print("🧠 开始遗传算法训练...")
        
        try:
            from toolkit.setupGPlearn import my_gplearn
            
            self.model = my_gplearn(
                self.function_set,
                stable_score_func,
                feature_names=selected_factors,
                pop_num=200,    # 减少种群数量，避免过拟合
                gen_num=3,      # 减少代数，避免过拟合
                random_state=42,
                verbose=0       # 减少输出
            )
            
            self.model.fit(x_array_train, y_target_valid)
            self.factor_expression = str(self.model)
            
            print(f"✅ 训练成功完成!")
            print(f"📝 挖掘的因子: {self.factor_expression}")
            
            return True
            
        except Exception as e:
            print(f"❌ 训练过程中出现错误: {e}")
            return False
    
    def predict_with_smart_volume_enhancement(self, price_df, x_dict, top_n=10):
        """智能成交量增强的预测逻辑 - 严格使用历史数据"""
        print(f"\n🔮 生成股票推荐 (智能成交量增强)...")
        
        # 🔧 关键修复：只使用训练数据的最后一天进行预测
        end_date = price_df.index[-1]
        recent_dates = [end_date]  # 只使用最后一天
        
        # 如果模型训练成功，使用模型预测
        if self.model is not None:
            try:
                # 准备预测数据 - 只使用最后一天
                recent_features = {}
                feature_names = list(x_dict.keys())[:12]  # 使用前12个因子
                
                for key in feature_names:
                    recent_feat = x_dict[key].loc[recent_dates]
                    recent_feat = recent_feat.reindex(columns=price_df.columns)
                    recent_features[key] = recent_feat.ffill().bfill().fillna(0)
                
                # 构建特征数组
                x_array_list = []
                for key in feature_names:
                    x_array_list.append(recent_features[key].values)
                
                recent_x_array = np.array(x_array_list)
                recent_x_array = np.transpose(recent_x_array, axes=(1, 2, 0))
                
                # 使用最后一天数据进行预测
                latest_features = recent_x_array[-1:, :, :]
                predictions = self.model.predict(latest_features)[0]
                
                print(f"✅ 模型预测成功!")
                
                # 创建股票得分序列
                stock_scores = pd.Series(predictions, index=price_df.columns)
                valid_scores = stock_scores.replace([np.inf, -np.inf], np.nan).dropna()
                
                if len(valid_scores) == 0:
                    print("⚠️  模型预测结果全为无效值，使用备选方案")
                    return self.volume_based_fallback(price_df, x_dict, top_n)
                
                stock_scores_std = (valid_scores - valid_scores.mean()) / valid_scores.std()
                stock_scores_std = stock_scores_std.fillna(0)
                
                # 使用智能成交量增强
                enhanced_scores = self.smart_volume_enhancement(stock_scores_std, recent_features)
                final_stocks = enhanced_scores.nlargest(top_n)
                
                return final_stocks
                
            except Exception as e:
                print(f"❌ 模型预测失败: {e}")
        
        # 如果模型预测失败，使用成交量备选方案
        print("🔄 切换到成交量备选方案...")
        return self.volume_based_fallback(price_df, x_dict, top_n)
    
    def smart_volume_enhancement(self, stock_scores, recent_features):
        """智能成交量增强"""
        print("🎯 智能成交量增强...")
        
        # 计算成交量得分
        volume_scores = pd.Series(0.0, index=stock_scores.index)
        
        for stock in stock_scores.index:
            volume_score = 0
            count = 0
            
            # 使用多个成交量因子
            volume_indicators = [
                ('VOLUME_RATIO_5', 1.0),
                ('VOLUME_RATIO_10', 0.8),
                ('VOLUME_RATIO_20', 0.6),
                ('VOLUME_BREAK_5', 0.5),
                ('VOLUME_BREAK_10', 0.3),
                ('VOLUME_BREAK_20', 0.2)
            ]
            
            for indicator, weight in volume_indicators:
                # 尝试不同大小写
                indicator_variants = [indicator, indicator.lower(), indicator.upper()]
                found = False
                
                for variant in indicator_variants:
                    if variant in recent_features and stock in recent_features[variant].columns:
                        value = recent_features[variant].iloc[-1][stock]
                        if not np.isnan(value):
                            if 'RATIO' in indicator.upper():
                                volume_score += min(value, 3.0) * weight
                            else:  # BREAK指标
                                if value:
                                    volume_score += 1.0 * weight
                            count += weight
                            found = True
                            break
                
                if not found:
                    # 如果找不到精确匹配，尝试包含关键词
                    for factor_name in recent_features:
                        if indicator.split('_')[0].lower() in factor_name.lower():
                            if stock in recent_features[factor_name].columns:
                                value = recent_features[factor_name].iloc[-1][stock]
                                if not np.isnan(value):
                                    if 'RATIO' in indicator.upper():
                                        volume_score += min(value, 3.0) * weight
                                    else:
                                        if value:
                                            volume_score += 1.0 * weight
                                    count += weight
                                    break
            
            if count > 0:
                volume_scores[stock] = volume_score / count
        
        # 标准化成交量得分
        if volume_scores.std() > 0:
            volume_scores_std = (volume_scores - volume_scores.mean()) / volume_scores.std()
        else:
            volume_scores_std = volume_scores * 0  # 全为0
        
        volume_scores_std = volume_scores_std.fillna(0)
        
        # 动态调整权重
        volume_score_std = volume_scores.std()
        if volume_score_std < 0.1:  # 如果成交量得分差异很小
            volume_weight = 0.15  # 降低成交量权重
            print(f"   成交量差异小，权重调整为: {volume_weight}")
        else:
            volume_weight = 0.3  # 正常权重
            print(f"   使用正常成交量权重: {volume_weight}")
        
        # 合并得分
        combined_scores = stock_scores * (1 - volume_weight) + volume_scores_std * volume_weight
        
        return combined_scores
    
    def volume_based_fallback(self, price_df, x_dict, top_n=10):
        """纯成交量策略备选"""
        print("🔄 使用纯成交量策略选股...")
        
        # 获取最新的成交量数据
        volume_ratios = None
        
        # 尝试不同的成交量因子名称
        volume_factor_candidates = [
            'VOLUME_RATIO_20', 'volume_ratio_20', 'VOL_RATIO_20', 'vol_ratio_20',
            'VOLUME_RATIO_10', 'volume_ratio_10', 'VOL_RATIO_10', 'vol_ratio_10',
            'VOLUME_RATIO_5', 'volume_ratio_5', 'VOL_RATIO_5', 'vol_ratio_5', 
            'vol'
        ]
        
        for factor_name in volume_factor_candidates:
            if factor_name in x_dict:
                volume_ratios = x_dict[factor_name].iloc[-1]
                print(f"   使用成交量因子: {factor_name}")
                break
        
        # 如果找不到成交量比率，计算一个简单的20日成交量比率
        if volume_ratios is None and 'vol' in x_dict:
            volume_data = x_dict['vol']
            if len(volume_data) >= 20:
                volume_ratios = volume_data.iloc[-1] / volume_data.rolling(20).mean().iloc[-1]
                print("   计算20日成交量比率")
        
        if volume_ratios is None:
            # 最后备选：随机选择
            import random
            stocks = list(price_df.columns)
            selected = random.sample(stocks, min(top_n, len(stocks)))
            return pd.Series([1.0] * len(selected), index=selected)
        
        # 清理数据
        volume_scores = volume_ratios.replace([np.isinf, -np.inf], np.nan).fillna(0)
        
        # 进一步要求正收益
        if 'pct_chg' in x_dict and len(x_dict['pct_chg']) >= 20:
            recent_returns = x_dict['pct_chg'].iloc[-20:].mean(axis=0)
            positive_momentum = recent_returns > 0
            final_stocks = volume_scores[volume_scores.index.isin(positive_momentum[positive_momentum].index)]
        else:
            final_stocks = volume_scores
        
        print(f"✅ 成交量策略选出 {len(final_stocks)} 只股票")
        return final_stocks.nlargest(top_n)

def main():
    """主函数 - 运行完整的回测（20日调仓周期）"""
    print("="*70)
    print("🎯 智能成交量增强策略回测系统 (20日调仓周期)")
    print("="*70)
    
    # 创建预测器和回测器
    predictor = SmartVolumeEnhancedPredictor()
    backtester = SmartVolumeBacktester(
        initial_capital=1000000,  # 初始资金100万
        transaction_cost=0.001    # 交易成本0.1%
    )
    
    # 1. 加载数据
    try:
        price_df, x_dict = predictor.load_and_prepare_data(
            price_file='close_price_800stocks_1year.pkl',
            factor_file='stock_factors_800stocks_1year.pkl'
        )
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return
    
    # 2. 运行回测 - 20日调仓周期
    performance = backtester.run_backtest(
        price_df=price_df,
        x_dict=x_dict,
        rebalance_days=20,       # 每20天调仓
        top_n=10,                # 每次选10只股票
        lookback_months=6        # 使用6个月历史数据训练
    )
    
    # 3. 生成报告和图表
    backtester.generate_report()
    backtester.plot_backtest_results()
    
    # 4. 保存结果
    backtester.save_backtest_results()
    
    print("\n🎉 回测完成!")

if __name__ == '__main__':
    main()