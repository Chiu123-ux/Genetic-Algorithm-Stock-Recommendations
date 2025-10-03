# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from scipy.stats import zscore
import dill
from datetime import datetime

class FutureStockPredictor:
    """
    使用历史数据直接预测推荐股票
    """
    
    def __init__(self):
        self.function_set = ['add', 'sub', 'mul', 'div', 'sqrt', 'log', 'abs', 'neg', 'inv', 'max', 'min'
                        , 'gtpn', 'andpn', 'orpn', 'ltpn', 'gtp', 'andp', 'orp', 'ltp', 'gtn',
                          'andn', 'orn', 'ltn', 'delayy', 'delta', 'signedpower', 'decayl', 'stdd', 'rankk']
        self.model = None
        self.factor_expression = None
        # 新增属性
        self.final_raw_ic = 0  # 原始IC值
        self.final_abs_ic = 0   # 绝对IC值
        self.min_acceptable_ic = 0.03  # IC值阈值
    
    def load_and_prepare_data(self, price_file, factor_file):
        """加载价格和因子数据"""
        print("📂 加载历史数据...")
        
        with open(price_file, 'rb') as f:
            price_df = dill.load(f)
        price_df.index = pd.to_datetime(price_df.index)
        
        with open(factor_file, 'rb') as f:
            x_dict = dill.load(f)
        for key in x_dict:
            x_dict[key].index = pd.to_datetime(x_dict[key].index)
            
        print(f"✅ 数据加载完成")
        print(f"   时间范围: {price_df.index[0].strftime('%Y-%m-%d')} 至 {price_df.index[-1].strftime('%Y-%m-%d')}")
        print(f"   股票数量: {len(price_df.columns)}")
        print(f"   因子数量: {len(x_dict)}")
        
        return price_df, x_dict
    
    def train_with_historical_data(self, price_df, x_dict, train_months=6, predict_days=10):
        """使用历史数据训练预测模型"""
        print(f"\n🎯 开始训练模型...")
        print(f"   训练周期: 过去{train_months}个月")
        print(f"   预测目标: 未来{predict_days}天表现")
        
        # 确定训练窗口
        end_date = price_df.index[-1]
        start_date = end_date - pd.DateOffset(months=train_months)
        
        # 准备训练数据
        price_train = price_df.loc[start_date:end_date]
        
        # 计算未来收益率作为训练目标
        future_returns = []
        valid_dates = []
        
        for i in range(len(price_train) - predict_days):
            current_date = price_train.index[i]
            future_date = price_train.index[i + predict_days]
            
            daily_returns = {}
            for stock in price_train.columns:
                current_price = price_train.iloc[i][stock]
                future_price = price_train.iloc[i + predict_days][stock]
                
                if not np.isnan(current_price) and not np.isnan(future_price) and current_price > 0:
                    future_ret = (future_price - current_price) / current_price
                    daily_returns[stock] = future_ret
            
            future_returns.append(daily_returns)
            valid_dates.append(current_date)
        
        # 创建目标变量DataFrame
        y_target = pd.DataFrame(future_returns, index=valid_dates)
        
        # 准备特征数据
        x_dict_train = {}
        for key in x_dict:
            feat_df = x_dict[key].loc[start_date:end_date].reindex_like(price_train)
            x_dict_train[key] = feat_df.ffill().bfill().fillna(0)
        
        feature_names = list(x_dict_train.keys())
        
        # 构建训练数据
        x_array_train = np.array([x_dict_train[key].values for key in feature_names])
        x_array_train = np.transpose(x_array_train, axes=(1, 2, 0))
        
        # 筛选有效日期
        date_mask = [date in valid_dates for date in price_train.index]
        x_array_valid = x_array_train[date_mask]
        y_target_valid = y_target.values
        
        # 修正的评分函数 - 优化绝对IC值
        def score_func(y, y_pred, sample_weight):
            if len(np.unique(y_pred[-1])) <= 10:
                return -1
            
            # 计算每日IC值
            daily_ics = []
            for day_idx in range(len(y)):
                y_day = y[day_idx]
                y_pred_day = y_pred[day_idx]
                
                # 创建DataFrame便于计算
                df_day = pd.DataFrame({'true': y_day, 'pred': y_pred_day})
                df_day = df_day.replace([np.inf, -np.inf], np.nan).dropna()
                
                if len(df_day) > 10:
                    ic_day = df_day['true'].corr(df_day['pred'], method='spearman')
                    if not np.isnan(ic_day):
                        daily_ics.append(abs(ic_day)) # 关键修改：使用绝对值
            
            if daily_ics:
                ic_mean = np.mean(daily_ics)  # 优化绝对IC均值
                return ic_mean
            else:
                return 0
        
        # 训练模型
        from toolkit.setupGPlearn import my_gplearn
        
        self.model = my_gplearn(
            self.function_set,
            score_func,
            feature_names=feature_names,
            pop_num=500,
            gen_num=5,
            random_state=42
        )
        
        self.model.fit(x_array_valid, y_target_valid)
        self.factor_expression = str(self.model)
        
        print(f"✅ 模型训练完成!")
        print(f"📝 挖掘的因子: {self.factor_expression}")
        
        # 计算最终因子IC值
        final_predictions = self.model.predict(x_array_valid)
        all_raw_ics = []
        all_abs_ics = []

        for day_idx in range(len(y_target_valid)):
            df_day = pd.DataFrame({
                'true': y_target_valid[day_idx], 
                'pred': final_predictions[day_idx]
            })
            df_day = df_day.replace([np.inf, -np.inf], np.nan).dropna()

            if len(df_day) > 10:
                ic_val = df_day['true'].corr(df_day['pred'], method='spearman')
                if not np.isnan(ic_val):
                    all_raw_ics.append(ic_val)
                    all_abs_ics.append(abs(ic_val))

        if all_raw_ics:
            self.final_raw_ic = np.mean(all_raw_ics)
            self.final_abs_ic = np.mean(all_abs_ics)
            print(f"📊 最终因子原始IC值: {self.final_raw_ic:.4f}")
            print(f"📈 最终因子绝对IC值: {self.final_abs_ic:.4f}")
            
            # 检查IC值是否达到阈值
            if self.final_abs_ic < self.min_acceptable_ic:
                print(f"⚠️  警告: 因子绝对IC值({self.final_abs_ic:.4f})低于阈值({self.min_acceptable_ic})")
                return False
        else:
            self.final_raw_ic = 0
            self.final_abs_ic = 0
            print(f"⚠️  无法计算有效的IC值")
            return False
        
        return True
    

    def predict_top_stocks(self, price_df, x_dict, top_n=10, recent_days=10):
        """预测推荐股票"""
        if self.model is None:
            print("❌ 请先训练模型!")
            return None
        
        print(f"\n🔮 生成股票推荐...")
        print(f"   使用最近{recent_days}天数据")
        print(f"   推荐数量: {top_n}只")
        
        # 获取最近数据
        end_date = price_df.index[-1]
        start_date = end_date - pd.DateOffset(days=recent_days)
        
        recent_dates = price_df.loc[start_date:end_date].index
        
        # 准备预测数据
        recent_features = {}
        for key in x_dict:
            recent_feat = x_dict[key].loc[recent_dates]
            # 重新索引以确保与价格数据对齐
            recent_feat = recent_feat.reindex(columns=price_df.columns)
            recent_features[key] = recent_feat.ffill().bfill().fillna(0)
        
        feature_names = list(recent_features.keys())
        
        # 构建特征数组
        x_array_list = []
        for key in feature_names:
            x_array_list.append(recent_features[key].values)
        
        recent_x_array = np.array(x_array_list)  # 形状: (因子, 时间, 股票)
        
        # 转置为 (时间, 股票, 因子)
        recent_x_array = np.transpose(recent_x_array, axes=(1, 2, 0))
        
        print(f"   预测数据形状: {recent_x_array.shape}")
        
        # 方法1: 使用最后一天数据进行预测
        latest_features = recent_x_array[-1:, :, :]  # 取最后一天
        
        try:
            predictions = self.model.predict(latest_features)[0]
            print(f"✅ 预测成功! 预测结果长度: {len(predictions)}")
            
            # 创建股票得分序列
            stock_scores = pd.Series(predictions, index=price_df.columns)
            valid_scores = stock_scores.replace([np.inf, -np.inf], np.nan).dropna()
            stock_scores_std = (valid_scores - valid_scores.mean()) / valid_scores.std()
            stock_scores_std = stock_scores_std.fillna(0)
            
            # 关键修改：统一选择得分最高的股票
            top_stocks = stock_scores_std.sort_values(ascending=False).head(top_n)
            print("✅ 使用正向选择（高分股票）")
            
            return top_stocks
            
        except Exception as e:
            print(f"❌ 单日预测失败: {e}")
            
            # 方法2: 使用多天平均
            print("🔄 尝试多天平均预测...")
            try:
                daily_predictions = []
                for i in range(len(recent_x_array)):
                    day_pred = self.model.predict(recent_x_array[i:i+1])[0]
                    daily_predictions.append(day_pred)
                
                avg_predictions = np.mean(daily_predictions, axis=0)
                stock_scores = pd.Series(avg_predictions, index=price_df.columns)
                valid_scores = stock_scores.replace([np.inf, -np.inf], np.nan).dropna()
                stock_scores_std = (valid_scores - valid_scores.mean()) / valid_scores.std()
                stock_scores_std = stock_scores_std.fillna(0)
                
                # 统一选择得分最高的股票
                top_stocks = stock_scores_std.sort_values(ascending=False).head(top_n)
                print("✅ 多天平均预测成功！使用正向选择")
                
                return top_stocks
                
            except Exception as e2:
                print(f"❌ 所有预测方法都失败了: {e2}")
                
                # 方法3: 直接使用因子值
                print("🔄 使用原始因子值...")
                latest_factor = recent_features['amount'].iloc[-1]  # 使用amount因子
                factor_scores = pd.Series(latest_factor, index=price_df.columns)
                factor_scores_std = zscore(factor_scores, nan_policy='omit').fillna(0)
                
                # 统一选择得分最高的股票
                top_stocks = factor_scores_std.sort_values(ascending=False).head(top_n)
                print("✅ 使用原始因子值成功！使用正向选择")
                
                return top_stocks
    
    def save_recommendations(self, recommendations, price_df, filename=None):
        """保存推荐结果"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f'stock_recommendations_{timestamp}.csv'
        
        # 创建详细报告
        report_data = []
        latest_prices = price_df.iloc[-1]
        
        for rank, (stock, score) in enumerate(recommendations.items(), 1):
            current_price = latest_prices.get(stock, 0)
            report_data.append({
                '排名': rank,
                '股票代码': stock,
                '因子得分': round(score, 4),
                '最新价格': round(current_price, 2) if current_price > 0 else 0
            })
        
        report_df = pd.DataFrame(report_data)
        report_df.to_csv(filename, index=False, encoding='utf-8-sig')
        
        print(f"\n💾 推荐结果已保存到: {filename}")
        print(f"📊 IC值详情: 原始IC={self.final_raw_ic:.4f}, 绝对IC={self.final_abs_ic:.4f})")
        
        return report_df

def main():
    """主函数"""
    # 创建预测器
    predictor = FutureStockPredictor()
    
    # 1. 加载数据
    try:
        price_df, x_dict = predictor.load_and_prepare_data(
            price_file='stock_close_3years.pkl',
            factor_file='factor_3years.pkl'
        )
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return
    
    # 2. 训练模型
    try:
        success = predictor.train_with_historical_data(
            price_df=price_df,
            x_dict=x_dict,
            train_months=6,
            predict_days=10
        )
    except Exception as e:
        print(f"❌ 模型训练失败: {e}")
        return
    
    if not success:
        print("❌ 因子预测能力不足，建议重新训练")
        return
    
    # 3. 预测推荐股票
    recommendations = predictor.predict_top_stocks(
        price_df=price_df,
        x_dict=x_dict,
        top_n=10,
        recent_days=10
    )
    
    if recommendations is not None:
        # 显示推荐结果
        print("\n" + "="*60)
        print("🏆 股票推荐结果")
        print("="*60)
        print(f"预测日期: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print(f"使用的因子: {predictor.factor_expression}")
        print(f"因子IC值: 原始={predictor.final_raw_ic:.4f}, 绝对={predictor.final_abs_ic:.4f})")
        print(f"选择方式: 正向选择（高分股票）")
        print("-"*60)
        
        latest_prices = price_df.iloc[-1]
        
        for rank, (stock, score) in enumerate(recommendations.items(), 1):
            current_price = latest_prices.get(stock, 0)
            print(f"{rank:2d}. {stock:<12} | 得分: {score:7.4f} | 价格: {current_price:8.2f}")
        
        # 保存结果
        report_df = predictor.save_recommendations(recommendations, price_df)

if __name__ == '__main__':
    main()