本质是遗传算法应用时序函数，后续改进方向是加入时间切片函数
用原始因子合成遗传算法选择的因子，然后进行回测，月度周度换手均可

主程序在新回测，也可以进行预测，在新实盘
首先要爬数据，用因子三年获取和收盘价获取到两个pkl文件

The essence is the application of a genetic algorithm to time-series functions. Future improvements involve incorporating time-slicing functions.

Use original factors to synthesize factors selected by the genetic algorithm, followed by backtesting. Monthly or weekly turnover can be applied.

The main program is used for new backtesting and can also be utilized for forecasting in live trading.

First, data must be scraped. Obtain two pickle files: one for factor data over three years and another for closing prices.
