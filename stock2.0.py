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
        # 检查全时段因子值是否恒定
        if df_pred.nunique(axis=1).min() == 1:
            logging.warning("全局因子值为常量，评分终止")
            return -1.0

        # 检查全时段唯一值分组能力
        global_unique = df_pred.nunique(axis=1).mean()
        if global_unique <= 10:
            return -1.0  # 分组能力不足

        # ------------------- 滚动窗口计算 -------------------
        roll_ic = []
        window_size = 21  # 月度滚动窗口

        for i in range(0, len(y), window_size):
            # 切片数据
            pred_slice = df_pred.iloc[i:i + window_size]
            y_slice = df_y.iloc[i:i + window_size]

            # 窗口级常量检查
            if pred_slice.nunique(axis=1).min() == 1 or y_slice.nunique(axis=1).min() == 1:
                logging.debug(f"窗口 {i}-{i + window_size} 存在常量输入，跳过")
                continue  # 跳过无效窗口

            # 计算窗口内每日Rank IC
            try:
                window_corr = y_slice.corrwith(pred_slice, axis=1, method='spearman')
                roll_ic.append(window_corr.mean())
            except Exception as e:
                logging.warning(f"窗口 {i} 计算失败: {str(e)}")
                continue

        # ------------------- IC合成 -------------------
        if len(roll_ic) == 0:  # 所有窗口均无效
            logging.warning("所有滚动窗口计算失败")
            return -1.0

        final_ic = np.nanmean(roll_ic)
        return final_ic if not np.isnan(final_ic) else 0.0

    except Exception as e:
        logging.error(f"评分函数崩溃: {str(e)}", exc_info=True)
        return 0.0

def preprocess_data(price_data):
    """数据预处理：缺失值处理+收益率计算+有效列标记"""
    # 填充缺失值（先向前后向后）
    price = price_data.ffill().bfill()

    # 计算未来收益率（5日和10日）
    ret5 = price.pct_change(5).shift(-5)
    ret10 = price.pct_change(10).shift(-10)

    # 剔除停牌数据（收益率缺失超过5天的股票）
    valid_stocks = ret5.isnull().mean() < 0.2
    valid_columns = price_data.columns[valid_stocks]  # 新增：有效列名

    # 返回筛选后的数据和有效列名
    return (
        price.loc[:, valid_columns],
        ret5.loc[:, valid_columns],
        ret10.loc[:, valid_columns],
        valid_columns  # 新增返回有效列名
    )

if __name__ == '__main__':
    try:
        # ---- 数据加载与预处理 ----
        logging.info(f"Process started at {datetime.now()}")
        with open('./data/stock_data.pickle', 'rb') as f:
            raw_price = dill.load(f)

        # 解包新增的valid_columns
        price, ret5, ret10, valid_columns = preprocess_data(raw_price)  # 修改点1/4
        logging.info(f"Data loaded: {price.shape}, valid stocks: {len(valid_columns)}")

        # ---- 特征标准化 ----
        with open('./data/factor_data.pickle', 'rb') as f:
            x_dict = dill.load(f)

        # 仅加载有效列对应的特征数据 修改点2/4
        feature_names = list(x_dict.keys())
        x_array = np.array([x_dict[k].loc[:, valid_columns].values for k in feature_names])  # 筛选有效列
        x_array = np.transpose(x_array, axes=(1, 2, 0))  # (time, stock, features)

        # 三维数据标准化
        scaler = StandardScaler()
        x_flat = x_array.reshape(-1, x_array.shape[-1])
        x_scaled = scaler.fit_transform(x_flat).reshape(x_array.shape)

        # 维度一致性检查 修改点3/4
        assert x_scaled.shape[1] == len(valid_columns), \
            f"特征/价格维度不匹配: 特征 {x_scaled.shape[1]}列 vs 价格 {len(valid_columns)}列"

        # ---- 遗传编程配置 ----
        function_set = [
            'add', 'sub', 'mul', 'div', 'sqrt', 'log',
            'abs', 'neg', 'inv', 'max', 'min'
        ]

        model_params = {
            'score_func_basic': robust_score_func,
            'feature_names': feature_names,
            'pop_num': 50,
            'gen_num': 5,
            'tour_num': 20,
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
        y_pred = pd.DataFrame(y_pred, index=price.index, columns=valid_columns)  # 修改点4/4

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