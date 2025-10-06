# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from scipy.stats import zscore
import dill
from datetime import datetime

class SmartVolumeEnhancedPredictor:
    """
    智能成交量增强的股票预测器 - 解决过滤过严问题
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
            volume_factors = [f for f in x_dict.keys() if 'VOLUME' in f]
            print(f"   📈 成交量因子: {len(volume_factors)}个")
            
            return price_df, x_dict
            
        except Exception as e:
            print(f"❌ 数据加载失败: {e}")
            raise
    
   
    def smart_volume_enhancement(self, stock_scores, recent_features):
        """智能成交量增强 - 不过滤，只加权"""
        print("🎯 智能成交量增强...")
        
        # 计算成交量得分（不过滤）
        volume_scores = pd.Series(0.0, index=stock_scores.index)
        
        # 调试信息：统计成交量比率分布
        volume_ratios_5 = []
        
        for stock in stock_scores.index:
            volume_score = 0
            count = 0
            
            # 使用多个成交量因子
            volume_indicators = [
                ('VOLUME_RATIO_5', 1.0),
                ('VOLUME_RATIO_10', 0.8),
                ('VOLUME_BREAK_5', 0.5),
                ('VOLUME_BREAK_10', 0.3)
            ]
            
            for indicator, weight in volume_indicators:
                if indicator in recent_features and stock in recent_features[indicator].columns:
                    value = recent_features[indicator].iloc[-1][stock]
                    if not np.isnan(value):
                        if 'RATIO' in indicator:
                            volume_ratios_5.append(value)
                            volume_score += min(value, 3.0) * weight
                        else:  # BREAK指标
                            if value:
                                volume_score += 1.0 * weight
                        count += weight
            
            if count > 0:
                volume_scores[stock] = volume_score / count
        
        # 输出成交量统计信息
        if volume_ratios_5:
            print(f"   📊 成交量比率统计:")
            print(f"      均值: {np.mean(volume_ratios_5):.3f}")
            print(f"      中位数: {np.median(volume_ratios_5):.3f}")
            print(f"      最大值: {np.max(volume_ratios_5):.3f}")
            print(f"      最小值: {np.min(volume_ratios_5):.3f}")
        
        # 标准化成交量得分
        volume_scores_std = (volume_scores - volume_scores.mean()) / volume_scores.std()
        volume_scores_std = volume_scores_std.fillna(0)
        
        # 动态调整权重：基于成交量得分的分布
        volume_score_std = volume_scores.std()
        if volume_score_std < 0.1:  # 如果成交量得分差异很小
            volume_weight = 0.2  # 降低成交量权重
            print(f"   成交量差异小，权重调整为: {volume_weight}")
        else:
            volume_weight = 0.4  # 正常权重
            print(f"   使用正常成交量权重: {volume_weight}")
        
        # 合并得分
        combined_scores = stock_scores * (1 - volume_weight) + volume_scores_std * volume_weight
        
        # 分析选股特征
        self.analyze_enhancement_results(stock_scores, volume_scores, combined_scores)
        
        return combined_scores
    
    def analyze_enhancement_results(self, original_scores, volume_scores, enhanced_scores):
        """分析增强效果"""
        print("\n📊 增强效果分析:")
        
        # 原始排名前10
        original_top = original_scores.nlargest(10)
        
        # 增强后排名前10
        enhanced_top = enhanced_scores.nlargest(10)
        
        # 计算排名变化
        common_stocks = set(original_top.index) & set(enhanced_top.index)
        print(f"   共同入选股票: {len(common_stocks)}只")
        
        # 分析成交量特征
        if len(volume_scores) > 0:
            avg_volume_score = volume_scores.mean()
            enhanced_avg_volume = volume_scores[enhanced_top.index].mean()
            print(f"   平均成交量得分: {avg_volume_score:.3f}")
            print(f"   增强后选股平均成交量得分: {enhanced_avg_volume:.3f}")
            
            if enhanced_avg_volume > avg_volume_score:
                print("   ✅ 成功提升了选股的成交量特征")
            else:
                print("   ⚠️  选股成交量特征未明显提升")
    
    def predict_with_smart_volume_enhancement(self, price_df, x_dict, top_n=10):
        """智能成交量增强的预测逻辑"""
        print(f"\n🔮 生成股票推荐 (智能成交量增强)...")
        
        # 获取最近数据
        end_date = price_df.index[-1]
        start_date = end_date - pd.DateOffset(days=5)
        recent_dates = price_df.loc[start_date:end_date].index
        
        # 如果模型训练成功，使用模型预测
        if self.model is not None:
            try:
                # 准备预测数据
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
                
                # 使用智能成交量增强（不过滤）
                enhanced_scores = self.smart_volume_enhancement(stock_scores_std, recent_features)
                final_stocks = enhanced_scores.nlargest(top_n)
                
                return final_stocks
                
            except Exception as e:
                print(f"❌ 模型预测失败: {e}")
        
        # 如果模型预测失败，使用成交量备选方案
        print("🔄 切换到成交量备选方案...")
        return self.volume_based_fallback(price_df, x_dict, top_n)
    
    def volume_based_fallback(self, price_df, x_dict, top_n=10):
        """纯成交量策略备选"""
        print("🔄 使用纯成交量策略选股...")
        
        # 获取最新的成交量数据
        if 'VOLUME_RATIO_5' in x_dict:
            volume_ratios = x_dict['VOLUME_RATIO_5'].iloc[-1]
        elif 'vol' in x_dict:
            volume_data = x_dict['vol']
            volume_ratios = volume_data.iloc[-1] / volume_data.rolling(5).mean().iloc[-1]
        else:
            return None
        
        # 清理数据
        volume_scores = volume_ratios.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # 进一步要求正收益
        if 'pct_chg' in x_dict:
            recent_returns = x_dict['pct_chg'].iloc[-5:].mean(axis=0)
            positive_momentum = recent_returns > 0
            final_stocks = volume_scores[volume_scores.index.isin(positive_momentum[positive_momentum].index)]
        else:
            final_stocks = volume_scores
        
        print(f"✅ 成交量策略选出 {len(final_stocks)} 只股票")
        return final_stocks.nlargest(top_n)
    
    def save_recommendations(self, recommendations, price_df, filename=None):
        """保存推荐结果"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f'smart_volume_recommendations_{timestamp}.csv'
        
        # 创建详细报告
        report_data = []
        latest_prices = price_df.iloc[-1]
        
        for rank, (stock, score) in enumerate(recommendations.items(), 1):
            current_price = latest_prices.get(stock, 0)
            report_data.append({
                '排名': rank,
                '股票代码': stock,
                '综合得分': round(score, 4),
                '最新价格': round(current_price, 2) if current_price > 0 else 0
            })
        
        report_df = pd.DataFrame(report_data)
        report_df.to_csv(filename, index=False, encoding='utf-8-sig')
        
        print(f"\n💾 推荐结果已保存到: {filename}")
        return report_df

def main():
    """主函数"""
    # 创建预测器
    predictor = SmartVolumeEnhancedPredictor()
    
    # 1. 加载数据
    try:
        price_df, x_dict = predictor.load_and_prepare_data(
            price_file='close_price_800stocks_1year.pkl',
            factor_file='stock_factors_800stocks_1year.pkl'
        )
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return
    
    # 2. 使用修复版快速训练
    try:
        success = predictor.robust_quick_train(
            price_df=price_df,
            x_dict=x_dict,
            train_months=6,
            predict_days=10
        )
        
        if success:
            print("🎉 模型训练成功！")
        else:
            print("⚠️  模型训练失败，将使用成交量备选方案")
            
    except Exception as e:
        print(f"❌ 训练过程出现异常: {e}")
        success = False
    
    # 3. 预测推荐股票 - 使用智能成交量增强
    recommendations = predictor.predict_with_smart_volume_enhancement(
        price_df=price_df,
        x_dict=x_dict,
        top_n=10
    )
    
    if recommendations is not None:
        # 显示推荐结果
        print("\n" + "="*60)
        print("🏆 智能成交量增强选股结果")
        print("="*60)
        print(f"预测日期: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        if hasattr(predictor, 'factor_expression') and predictor.factor_expression:
            print(f"使用的因子: {predictor.factor_expression}")
        print("-"*60)
        
        latest_prices = price_df.iloc[-1]
        
        for rank, (stock, score) in enumerate(recommendations.items(), 1):
            current_price = latest_prices.get(stock, 0)
            print(f"{rank:2d}. {stock:<12} | 得分: {score:7.4f} | 价格: {current_price:8.2f}")
        
        # 保存结果
        predictor.save_recommendations(recommendations, price_df)

if __name__ == '__main__':

    main()

