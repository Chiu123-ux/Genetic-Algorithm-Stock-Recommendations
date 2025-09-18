import numpy as np
import pandas as pd
import tensorflow as tf
import os
import time
from tensorflow.contrib import rnn
from sklearn.preprocessing import StandardScaler
from typing import List

#%% #################### 增强型特征工程 ####################
class GeneticFeatureEngineer:
    def __init__(self, genetic_factors: List[str], window_size=60):
        self.genetic_factors = genetic_factors
        self.window_size = window_size
        self.scalers = {}
        
    def _add_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加遗传因子交互特征"""
        for i in range(len(self.genetic_factors)):
            for j in range(i+1, len(self.genetic_factors)):
                f1, f2 = self.genetic_factors[i], self.genetic_factors[j]
                df[f'{f1}_x_{f2}'] = df[f1] * df[f2]
                df[f'{f1}_div_{f2}'] = df[f1] / (df[f2] + 1e-6)
        return df

    def process(self, stock_data: pd.DataFrame) -> np.ndarray:
        """动态滚动标准化处理"""
        # 合并基础特征和遗传因子
        features = stock_data[['open_hfq','high_hfq','low_hfq','close_hfq'] + self.genetic_factors]
        
        # 添加交互特征
        features = self._add_interaction_features(features)
        
        # 滚动标准化
        for col in features.columns:
            roll_mean = features[col].rolling(window=self.window_size, min_periods=1).mean()
            roll_std = features[col].rolling(window=self.window_size, min_periods=1).std() + 1e-6
            features[col] = (features[col] - roll_mean) / roll_std
            
        return self._create_sequences(features.values)
    
    def _create_sequences(self, data: np.ndarray, window=60) -> np.ndarray:
        """创建时间序列样本"""
        sequences = []
        for i in range(len(data)-window):
            seq = data[i:i+window]
            sequences.append(seq)
        return np.array(sequences)

#%% #################### 优化后的模型架构 ####################  
class EnhancedLSTMModel:
    def __init__(self, input_shape, num_factors, layer_num=3, cell_num=512):
        self.inputs = tf.placeholder(tf.float32, [None, input_shape[0]*input_shape[1]])
        self.targets = tf.placeholder(tf.float32, [None, 1])
        self.keep_prob = tf.placeholder(tf.float32)
        
        # 动态调整输入维度
        adjusted_input_dim = input_shape[1] + num_factors
        
        # 增强的输入层
        with tf.variable_scope('EnhancedInput'):
            x = tf.reshape(self.inputs, [-1, input_shape[0], adjusted_input_dim])
            x = tf.layers.dense(x, 256, activation=tf.nn.elu)
            x = tf.layers.dropout(x, rate=self.keep_prob)
            
        # 多尺度LSTM层
        with tf.variable_scope('MultiScaleLSTM'):
            cell = tf.nn.rnn_cell.MultiRNNCell([
                self._build_lstm_cell(cell_num//(2**i)) 
                for i in range(layer_num)
            ])
            outputs, state = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)
            last_output = self._attention_layer(outputs)
            
        # 残差输出层
        with tf.variable_scope('ResidualOutput'):
            dense1 = tf.layers.dense(last_output, 128, activation=tf.nn.leaky_relu)
            dense2 = tf.layers.dense(dense1, 64, activation=tf.nn.leaky_relu)
            self.predictions = tf.layers.dense(dense2, 1)
            
        # 自适应损失函数
        self.loss = self._sharpe_aware_loss()
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(self.loss)
        
    def _build_lstm_cell(self, units):
        cell = rnn.LSTMCell(units)
        return rnn.DropoutWrapper(cell, output_keep_prob=self.keep_prob)
    
    def _attention_layer(self, inputs):
        query = tf.layers.dense(inputs, 64, activation=tf.nn.tanh)
        keys = tf.layers.dense(inputs, 64, activation=tf.nn.tanh)
        attention = tf.nn.softmax(tf.matmul(query, keys, transpose_b=True))
        return tf.reduce_sum(attention * inputs, axis=1)
    
    def _sharpe_aware_loss(self):
        """基于夏普率的自适应损失函数"""
        returns = self.predictions - self.targets
        excess_returns = returns - tf.reduce_mean(returns)
        stddev = tf.math.reduce_std(excess_returns) + 1e-6
        sharpe_ratio = tf.reduce_mean(excess_returns) / stddev
        return tf.reduce_mean(tf.square(returns)) - 0.1 * sharpe_ratio

#%% #################### 增强的训练流程 ####################
class EnhancedTradingSystem:
    def __init__(self, genetic_factors=['turnover', 'volume', 'cir_market_value']):
        self.feature_engine = GeneticFeatureEngineer(genetic_factors)
        self.model = None
        self.data_processor = get_stock_data()
        
    def prepare_data(self, raw_path):
        """增强数据预处理流程"""
        # 原始数据处理
        self.data_processor.make_train_test_csv(orgin_data_path=raw_path)
        
        # 遗传因子增强处理
        raw_data = pd.read_csv(raw_path)
        genetic_data = self.feature_engine.process(raw_data)
        return genetic_data
    
    def build_model(self, input_shape, num_factors):
        """构建混合模型"""
        self.model = EnhancedLSTMModel(input_shape, num_factors)
        
    def train(self, train_data, epochs=100, batch_size=256):
        """增强的训练流程"""
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            
            # 动态学习率衰减
            lr = 0.001
            for epoch in range(epochs):
                # 学习率衰减
                if epoch % 20 == 0 and epoch != 0:
                    lr *= 0.5
                
                try:
                    x_batch, y_batch = self.data_processor.get_train_test_data_new(
                        batch_size, train_data)
                    
                    # 添加随机噪声增强数据
                    noise = np.random.normal(scale=0.01, size=x_batch.shape)
                    x_batch += noise
                    
                    feed_dict = {
                        self.model.inputs: x_batch,
                        self.model.targets: y_batch,
                        self.model.keep_prob: 0.6
                    }
                    
                    _, loss = sess.run([self.model.optimizer, self.model.loss], 
                                     feed_dict=feed_dict)
                    
                    if epoch % 10 == 0:
                        print(f"Epoch {epoch}, Loss: {loss:.4f}")
                        
                except StopIteration:
                    print("Training completed")
                    break
                
            # 保存完整模型
            saver = tf.train.Saver()
            saver.save(sess, "enhanced_stock_model.ckpt")
            
    def evaluate(self, test_data):
        """增强的评估流程"""
        with tf.Session() as sess:
            saver = tf.train.Saver()
            saver.restore(sess, "enhanced_stock_model.ckpt")
            
            x_test, y_test = self.data_processor.get_train_test_data_new(
                1000, test_data)
            
            predictions = sess.run(self.model.predictions, 
                                 feed_dict={
                                     self.model.inputs: x_test,
                                     self.model.keep_prob: 1.0
                                 })
            
            # 计算夏普率
            returns = predictions.flatten() - y_test.flatten()
            sharpe = np.mean(returns) / (np.std(returns) + 1e-6)
            print(f"Test Sharpe Ratio: {sharpe:.4f}")
            
#%% #################### 执行入口 ####################            
if __name__ == "__main__":
    # 初始化增强交易系统
    trading_system = EnhancedTradingSystem()
    
    # 数据准备阶段
    raw_data_path = "origin_data.csv"
    processed_data = trading_system.prepare_data(raw_data_path)
    
    # 模型构建（输入维度需要根据实际特征数量调整）
    num_genetic_factors = 3  # 对应turnover/volume/cir_market_value三个因子
    time_steps = 60
    input_dim = processed_data.shape[2]  # 自动获取特征维度
    
    trading_system.build_model(input_shape=(time_steps, input_dim), 
                             num_factors=num_genetic_factors)
    
    # 训练模型
    trading_system.train("train_data.csv", epochs=200)
    
    # 评估模型
    trading_system.evaluate("test_data.csv")