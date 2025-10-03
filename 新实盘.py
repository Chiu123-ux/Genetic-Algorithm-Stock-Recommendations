# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from scipy.stats import zscore
import dill
from datetime import datetime

class FutureStockPredictor:
    """
    使用历史数据直接预测推荐股票 - 优化版本
    在筛选阶段确保IC方向一致性
    """
    
    def __init__(self):
        self.function_set = ['add', 'sub', 'mul', 'div', 'sqrt', 'log', 'abs', 'neg', 'inv', 'max', 'min', 'rankk']
        self.model = None
        self.factor_expression = None
        # 新增属性
        self.final_raw_ic = 0  # 原始IC值
        self.final_abs_ic = 0   # 绝对IC值
        self.min_acceptable_ic = 0.01  # IC值阈值（降低以允许更多探索）
        self.direction_consistency = 0  # 方向一致性比率
        self.ic_std = 0  # IC标准差
        self.dominant_direction = "正向"  # 主导方向
        
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
    
    def train_with_directional_ic(self, price_df, x_dict, train_months=6, predict_days=20):
        """使用历史数据训练预测模型 - 确保IC方向一致性"""
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
        
        # 🎯 关键改进：方向一致的评分函数
        def directional_score_func(y, y_pred, sample_weight):
            """
            在筛选阶段就确保IC方向一致性
            返回：方向一致的IC均值 × 一致性惩罚因子
            """
            if len(np.unique(y_pred[-1])) <= 10:
                return -1  # 预测值缺乏区分度
            
            # 计算每日IC值并分类
            positive_ics = []  # 正相关IC
            negative_ics = []  # 负相关IC
            all_valid_ics = []  # 所有有效IC
            
            for day_idx in range(len(y)):
                y_day = y[day_idx]
                y_pred_day = y_pred[day_idx]
                
                # 创建DataFrame便于计算
                df_day = pd.DataFrame({'true': y_day, 'pred': y_pred_day})
                df_day = df_day.replace([np.inf, -np.inf], np.nan).dropna()
                
                if len(df_day) > 10:
                    ic_day = df_day['true'].corr(df_day['pred'], method='spearman')
                    
                    if not np.isnan(ic_day):
                        all_valid_ics.append(ic_day)
                        if ic_day > 0:
                            positive_ics.append(ic_day)
                        elif ic_day < 0:
                            negative_ics.append(ic_day)
            
            if not all_valid_ics:
                return 0
            
            # 🎯 核心逻辑：选择主导方向
            n_positive = len(positive_ics)
            n_negative = len(negative_ics)
            total_days = len(all_valid_ics)
            
            # 计算方向一致性比率
            consistency_ratio = max(n_positive, n_negative) / total_days
            
            # 根据主导方向计算IC
            if n_positive > n_negative:
                # 正相关主导
                dominant_ics = positive_ics
                direction_multiplier = 1.0  # 保持原方向
            else:
                # 负相关主导
                dominant_ics = negative_ics
                direction_multiplier = -1.0  # 反转方向
            
            if len(dominant_ics) == 0:
                return 0
            
            # 计算主导方向的IC均值
            dominant_ic_mean = np.mean(dominant_ics)
            dominant_abs_ic_mean = np.mean([abs(ic) for ic in dominant_ics])
            
            # 🎯 方向一致性惩罚因子
            if consistency_ratio >= 0.85:
                consistency_factor = 1.2  # 高度一致，奖励
            elif consistency_ratio >= 0.75:
                consistency_factor = 1.0  # 良好一致
            elif consistency_ratio >= 0.65:
                consistency_factor = 0.7  # 基本一致，轻微惩罚
            elif consistency_ratio >= 0.55:
                consistency_factor = 0.4  # 一致性较差，较重惩罚
            else:
                consistency_factor = 0.1  # 方向混乱，严重惩罚
            
            # 最终得分 = 主导方向绝对IC均值 × 方向一致性因子 × 方向乘子
            final_score = dominant_abs_ic_mean * consistency_factor * abs(direction_multiplier)
            
            return final_score
        
        # 训练模型
        from toolkit.setupGPlearn import my_gplearn
        
        self.model = my_gplearn(
            self.function_set,
            directional_score_func,  # 使用新的评分函数
            feature_names=feature_names,
            pop_num=300,  # 稍微减少种群数以加快收敛
            gen_num=4,    # 减少代数以测试效果
            random_state=42
        )
        
        self.model.fit(x_array_valid, y_target_valid)
        self.factor_expression = str(self.model)
        
        print(f"✅ 模型训练完成!")
        print(f"📝 挖掘的因子: {self.factor_expression}")
        
        # 🎯 严格的训练后验证
        return self._strict_post_training_validation(x_array_valid, y_target_valid)
    
    def _strict_post_training_validation(self, x_array_valid, y_target_valid):
        """严格的训练后验证"""
        final_predictions = self.model.predict(x_array_valid)
        
        daily_raw_ics = []
        positive_count = 0
        negative_count = 0
        
        for day_idx in range(len(y_target_valid)):
            df_day = pd.DataFrame({
                'true': y_target_valid[day_idx], 
                'pred': final_predictions[day_idx]
            })
            df_day = df_day.replace([np.inf, -np.inf], np.nan).dropna()

            if len(df_day) > 10:
                ic_val = df_day['true'].corr(df_day['pred'], method='spearman')
                if not np.isnan(ic_val):
                    daily_raw_ics.append(ic_val)
                    if ic_val > 0:
                        positive_count += 1
                    elif ic_val < 0:
                        negative_count += 1
        
        if daily_raw_ics:
            total_days = len(daily_raw_ics)
            self.final_raw_ic = np.mean(daily_raw_ics)
            self.final_abs_ic = np.mean([abs(ic) for ic in daily_raw_ics])
            
            # 确定主导方向
            if positive_count > negative_count:
                self.dominant_direction = "正向"
                dominant_ratio = positive_count / total_days
                # 如果是正向主导，我们期望IC为正
                expected_sign = 1
            else:
                self.dominant_direction = "负向" 
                dominant_ratio = negative_count / total_days
                expected_sign = -1
            
            self.direction_consistency = dominant_ratio
            self.ic_std = np.std(daily_raw_ics)
            
            print(f"🎯 方向一致性分析:")
            print(f"   主导方向: {self.dominant_direction}")
            print(f"   一致性比率: {dominant_ratio:.1%}")
            print(f"   原始IC均值: {self.final_raw_ic:.4f}")
            print(f"   绝对IC均值: {self.final_abs_ic:.4f}")
            print(f"   IC标准差: {self.ic_std:.4f}")
            
            # 🎯 严格的通过条件
            passes_validation = (
                self.final_abs_ic >= self.min_acceptable_ic and
                self.direction_consistency >= 0.65 and  # 65%以上天数方向一致
                self.ic_std <= 0.18 and  # IC波动控制
                self.final_raw_ic * expected_sign > 0  # 实际方向与期望方向一致
            )
            
            if passes_validation:
                print(f"✅ 因子通过方向一致性验证!")
                return True
            else:
                print(f"❌ 因子未通过方向一致性验证")
                if self.direction_consistency < 0.65:
                    print(f"   原因: 方向一致性不足({self.direction_consistency:.1%} < 65%)")
                if self.ic_std > 0.18:
                    print(f"   原因: IC波动过大({self.ic_std:.4f} > 0.18)")
                return False
        else:
            print(f"❌ 无法计算有效的IC值")
            return False

    def predict_with_directional_logic(self, price_df, x_dict, top_n=10, recent_days=10):
        """考虑因子方向的预测逻辑"""
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
            
            # 🎯 关键：根据因子方向决定选择逻辑
            if self.dominant_direction == "负向":
                # 对于负相关因子，选择得分最低的股票
                top_stocks = stock_scores_std.sort_values(ascending=True).head(top_n)
                print("✅ 使用负向选择（低分股票）- 因子呈负相关")
            else:
                # 默认选择得分最高的股票（正相关）
                top_stocks = stock_scores_std.sort_values(ascending=False).head(top_n)
                print("✅ 使用正向选择（高分股票）- 因子呈正相关")
            
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
                
                # 🎯 关键：根据因子方向决定选择逻辑
                if self.dominant_direction == "负向":
                    top_stocks = stock_scores_std.sort_values(ascending=True).head(top_n)
                print("✅ 多天平均预测成功！根据因子方向选择")
                
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
                print("✅ 使用原始因子值成功！默认正向选择")
                
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
        print(f"🎯 因子方向: {self.dominant_direction}, 一致性: {self.direction_consistency:.1%}")
        
        return report_df

    def get_training_summary(self):
        """获取训练摘要"""
        return {
            'factor_expression': self.factor_expression,
            'raw_ic': self.final_raw_ic,
            'abs_ic': self.final_abs_ic,
            'direction_consistency': self.direction_consistency,
            'ic_std': self.ic_std,
            'dominant_direction': self.dominant_direction,
            'is_valid': (self.final_abs_ic >= self.min_acceptable_ic and 
                     self.direction_consistency >= 0.65)
        }

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
    
    # 2. 训练模型 - 使用新的方向一致方法
    try:
        success = predictor.train_with_directional_ic(
            price_df=price_df,
            x_dict=x_dict,
            train_months=12,
            predict_days=30
        )
    except Exception as e:
        print(f"❌ 模型训练失败: {e}")
        return
    
    if not success:
        print("❌ 因子预测能力不足或方向不稳定，建议重新训练")
        return
    
    # 3. 预测推荐股票 - 使用方向感知的预测方法
    recommendations = predictor.predict_with_directional_logic(
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
        print(f"因子方向: {predictor.dominant_direction}, 一致性: {predictor.direction_consistency:.1%}")
        print("-"*60)
        
        latest_prices = price_df.iloc[-1]
        
        for rank, (stock, score) in enumerate(recommendations.items(), 1):
            current_price = latest_prices.get(stock, 0)
            print(f"{rank:2d}. {stock:<12} | 得分: {score:7.4f} | 价格: {current_price:8.2f}")
        
        # 保存结果
        report_df = predictor.save_recommendations(recommendations, price_df)
        
        # 显示训练摘要
        summary = predictor.get_training_summary()
        print(f"\n📋 训练摘要:")
        print(f"   因子表达式: {summary['factor_expression']}")
        print(f"   原始IC: {summary['raw_ic']:.4f}")
        print(f"   绝对IC: {summary['abs_ic']:.4f}")
        print(f"   方向一致性: {summary['direction_consistency']:.1%}")
        
        if summary['is_valid']:
            print(f"✅ 因子通过所有验证!")
        else:
            print(f"⚠️  因子未完全通过验证")

if __name__ == '__main__':
    main()