import numpy as np
import pandas as pd
import dill
import logging
from datetime import datetime
from functools import wraps
from sklearn.preprocessing import StandardScaler
from toolkit.backtest import BackTester
from toolkit.DataProcess import load_selecting_data
from toolkit.setupGPlearn import gp_save_factor, my_gplearn
from toolkit.IC import get_ic, calculate_ic, layered_returns
import os
import warnings

# 屏蔽特定 FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning,
                        message="The behavior of array concatenation with empty entries is deprecated.")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in cast")

# 配置日志和路径
if not os.path.exists('./result/factor/'):
    os.makedirs('./result/factor/')
logging.basicConfig(filename='./result/factor/generation.log', level=logging.INFO)

def robust_score_func(y, y_pred, sample_weight):
    """增强版评分函数：全时段分组能力检查 + 滚动窗口IC计算 + 常量输入处理"""
    try:
        df_pred = pd.DataFrame(y_pred)
        df_y = pd.DataFrame(y)

        # ------------------- 全局检查 -------------------
        if df_pred.nunique(axis=1).min() == 1:
            logging.warning("全局因子值为常量，评分终止")
            return 0.0

        global_unique = df_pred.nunique(axis=1).mean()
        if global_unique <= 10:
            return 0.0

        # ------------------- 滚动窗口计算 -------------------
        roll_ic = []
        window_size = 21

        for i in range(0, len(y), window_size):
            pred_slice = df_pred.iloc[i:i + window_size]
            y_slice = df_y.iloc[i:i + window_size]

            if pred_slice.nunique(axis=1).min() == 1 or y_slice.nunique(axis=1).min() == 1:
                logging.debug(f"窗口 {i}-{i + window_size} 存在常量输入，跳过")
                continue

            try:
                window_corr = y_slice.corrwith(pred_slice, axis=1, method='spearman')
                roll_ic.append(window_corr.mean())
            except Exception as e:
                logging.warning(f"窗口 {i} 计算失败: {str(e)}")
                continue

        if len(roll_ic) == 0:
            logging.warning("所有滚动窗口计算失败")
            return 0.0

        final_ic = np.nanmean(roll_ic)
        return abs(final_ic) if not np.isnan(final_ic) else 0.0

    except Exception as e:
        logging.error(f"评分函数崩溃: {str(e)}", exc_info=True)
        return 0.0

def preprocess_data(price_data):
    """数据预处理：统一索引+列名+缺失值处理"""
    # 填充缺失值
    price = price_data.ffill().bfill()

    # 计算未来收益率（假设5日窗口导致首尾各5行无效）
    ret5 = price.pct_change(5).shift(-5)
    ret10 = price.pct_change(10).shift(-10)

    # 筛选有效列
    valid_mask = (
        (price.isnull().mean() == 0) &
        (ret5.isnull().mean() < 0.2) &
        (ret10.isnull().mean() < 0.2)
    )
    valid_columns = price.columns[valid_mask]

    # 计算有效行索引（根据特征处理逻辑调整）
    valid_start = 5    # 因pct_change(5)导致前5行无效
    valid_end = -5     # 因shift(-5)导致最后5行无效
    valid_index = price.index[valid_start:valid_end]

    # 统一输出数据
    return (
        price.loc[valid_index, valid_columns],
        ret5.loc[valid_index, valid_columns],
        ret10.loc[valid_index, valid_columns],
        valid_columns,
        valid_index  # 新增有效索引
    )

if __name__ == '__main__':
    try:
        # ---- 数据加载与预处理 ----
        logging.info(f"Process started at {datetime.now()}")
        with open("C:/Users/bu'zhi'h'h'h/Desktop/pycharm/遗传算法/gplearnplus/use-gplearn-to-generate-CTA-factor/stock_prices.pkl", 'rb') as f:
            raw_price = dill.load(f)

        # 获取预处理数据（含有效索引）
        price, ret5, ret10, valid_columns, valid_index = preprocess_data(raw_price)
        logging.info(f"Data loaded: {price.shape}, valid stocks: {len(valid_columns)}")

        # ---- 特征数据对齐 ----
        with open("C:/Users/bu'zhi'h'h'h/Desktop/pycharm/遗传算法/gplearnplus/use-gplearn-to-generate-CTA-factor/factor_time.pkl", 'rb') as f:
            x_dict = dill.load(f)

        # 获取所有特征和价格数据的公共日期索引
        all_dates = [set(df.index) for df in x_dict.values()]
        common_dates = set.intersection(*all_dates, set(valid_index))  # 与价格数据有效索引的交集
        common_dates = sorted(common_dates)  # 按时间排序

        # 检查公共日期是否为空
        if not common_dates:
            raise ValueError("特征数据与价格数据无公共日期，请检查时间范围是否一致")

        # 统一列名
        common_columns = list(set.intersection(*[set(df.columns) for df in x_dict.values()]))

        # 检查公共列名是否为空
        if not common_columns:
            raise ValueError("特征数据与价格数据无公共股票列名，请检查列名是否一致")

        # 强制对齐所有特征数据的日期和列名
        for key in x_dict:
            x_dict[key] = x_dict[key].reindex(index=common_dates, columns=common_columns)
            # 填充缺失值（向前填充 + 向后填充）
            x_dict[key] = x_dict[key].ffill().bfill()

        # 检查特征数据是否全为空
        for key in x_dict:
            if x_dict[key].isnull().all().all():
                raise ValueError(f"特征 {key} 在公共日期范围内全为 NaN")

        # 更新全局有效索引和列名
        valid_index = common_dates
        valid_columns = common_columns

        # 构建三维特征数组
        feature_names = list(x_dict.keys())
        x_array = np.array([x_dict[k].values for k in feature_names])  # 已对齐无需再筛选
        x_array = np.transpose(x_array, axes=(1, 2, 0))  # (time, stock, features)

        # 三维标准化
        scaler = StandardScaler()
        x_flat = x_array.reshape(-1, x_array.shape[-1])
        x_scaled = scaler.fit_transform(x_flat).reshape(x_array.shape)

        # 维度一致性验证
        assert x_scaled.shape[:2] == (len(valid_index), len(valid_columns)), \
            f"数据维度不匹配: 预期 ({len(valid_index)}, {len(valid_columns)}) 实际 {x_scaled.shape[:2]}"

        # ---- 遗传编程配置 ----
        function_set = ['add', 'sub', 'mul', 'div', 'sqrt', 'log', 'abs', 'neg', 'inv', 'max', 'min'
                        , 'gtpn', 'andpn', 'orpn', 'ltpn', 'gtp', 'andp', 'orp', 'ltp', 'gtn', 'andn', 'orn', 'ltn', 'delayy', 'delta', 'signedpower', 'decayl', 'stdd', 'rankk']

        model_params = {
            'score_func_basic': robust_score_func,
            'feature_names': feature_names,
            'pop_num': 20,
            'gen_num': 2,
            'tour_num': 5,
            'random_state': 42,
            'verbose': 1
        }

        # ---- 因子生成 ----
        cmodel = my_gplearn(
            function_set=function_set,
            score_func_basic=model_params['score_func_basic'],
            pop_num=model_params['pop_num'],
            gen_num=model_params['gen_num'],
            tour_num=model_params['tour_num'],
            feature_names=model_params['feature_names'],
            random_state=model_params['random_state'],
            verbose=model_params['verbose']
        )
        cmodel.fit(x_scaled, np.array(ret5))
        logging.info(f"Best program:\n{cmodel._program}")

        # ---- 因子评估 ----
        y_pred = cmodel.predict(x_scaled)
        y_pred = pd.DataFrame(y_pred, index=valid_index, columns=valid_columns)  # 使用统一索引

        IC_df, IC_stat = get_ic(y_pred, ret5, ret10)
        print(f"IC Statistics:\n{IC_stat.to_markdown()}")

        layered_returns(y_pred, ret5, quantiles=5, periods=[5, 10])

        # ---- 因子保存 ----
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        factor_id = f"F_{timestamp}_G{model_params['gen_num']}P{model_params['pop_num']}"
        gp_save_factor(cmodel, factor_id)
        logging.info(f"Factor {factor_id} saved successfully")

    except Exception as e:
        logging.critical(f"Critical error occurred: {str(e)}", exc_info=True)
        raise