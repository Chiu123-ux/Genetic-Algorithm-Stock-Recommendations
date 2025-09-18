import pandas as pd
import numpy as np
from scipy import stats
from toolkit.my_plot import my_plot 

def get_ic(factor, ret5, ret10, method = 'spearman', print_type = 2, factor_name = 1):
    IC_df = pd.DataFrame()
    IC_df['5日IC'] = calculate_ic(ret5, factor, method)
    IC_df['10日IC'] = calculate_ic(ret10, factor, method)
    IC_statistic = ic_summary(IC_df)
    print(IC_statistic)
    my_plot(IC_df.cumsum(),['IC累计图', '时间', 'IC累计值', f'{factor_name}因子IC图']).line_plot()
    return IC_df, IC_statistic
   
def calculate_ic(ret, factor, method='spearman'):
    return ret.corrwith(factor, axis=1, method=method)
 
def ic_summary(IC_df):
    statistics = {
        "IC mean": round(IC_df.mean(), 4),
        "IC std": round(IC_df.std(), 4),
        "IR": round(IC_df.mean() / IC_df.std(), 4),
        "IR_LAST_1Y": round(IC_df[-240:].mean() / IC_df[-240:].std(), 4),
        "IC>0": round(len(IC_df[IC_df > 0].dropna()) / len(IC_df), 4),
        "ABS_IC>2%": round(len(IC_df[abs(IC_df) > 0.02].dropna()) / len(IC_df), 4)
    }
    return pd.DataFrame(statistics)


def layered_returns(factor, returns, quantiles=5, periods=[5, 10]):
    layered_ret = pd.DataFrame()

    for p in periods:
        # 生成分位数标签（确保无重复）
        labels = factor.rank(axis=1, pct=True).apply(
            lambda x: pd.qcut(x, quantiles, labels=False, duplicates='drop'),
            axis=1
        )

        # 转换为1D Series并去重
        labels_series = labels.stack().reset_index(level=1, drop=True)
        labels_series = labels_series[~labels_series.index.duplicated()]  # 新增去重

        # 对齐收益率数据
        ret = returns.shift(-p).stack().reset_index(level=1, drop=True)

        # 分组计算
        grouped_ret = ret.groupby(labels_series).mean().reset_index()
        grouped_ret.columns = ['Quantile', f'{p}D Return']

        layered_ret = pd.concat([layered_ret, grouped_ret.set_index('Quantile')], axis=1)

    return layered_ret

if __name__ == '__main__':
    vwap = pd.read_csv('../data/daily_5/vwap.csv',index_col=[0],parse_dates=[0])
    price = vwap
    ret5 = np.log(price.shift(-5) / price.shift(-1))
    ret10 = np.log(price.shift(-10) / price.shift(-1))
    IC_df, IC_statistic = get_ic(vwap) # 自相关系数
