# 基于深度学习方法的股票价格波动预测

## python环境和模块

`python3 # crawl ticker list need python2`  
`pip install bs4` 
`pip install gensim`  
`pip install theano`  
`pip install yfinance`   
`pip install keras`  


## 操作步骤

### 1 准备数据

#### 1.1 从NASDAQ网站抓取股票代码信息 

```python
python src/crawler_tickers.py top_n tickers_file
python src/crawler_tickers.py 20 data/ticker_list.csv
```
注意: top_n表示市场容量为前top_n%，可以调整；这里要用python2 urllib2抓数据，因为尝试用python3 urllib3抓数据时会有网络错误

#### 1.2 从Reuters抓取股票代码相关的财经新闻

```python
python src/crawler_reuters.py tickers_file finished_tickers_file failed_tickers_file news_file
python src/crawler_reuters.py data/ticker_list.csv data/finished_tickers.csv data/news_failed_tickers.csv data/news_reuters.csv  
```
注意: 第一次抓新闻的时候，要确保finished_tickers_file文件为空，否则会过滤相关股票的新闻

#### 1.3 从Yahoo Finance抓取股票代码相关的价格信息

```python
python src/crawler_yahoo_finance.py news_file finished_tickers_file raw_prices_file
python src/crawler_yahoo_finance.py data/news_reuters.csv data/finished_tickers.csv data/stock_prices_raw.json
```
注意: 重新抓取价格信息时，请删除原有文件

#### 1.4 对抓取的价格信息进行处理，获取每只股票短期、中期、长期的价格变动（正数表示涨幅，负数代表跌幅）

```python
python src/create_label.py raw_prices_file final_prices_file
python src/create_label.py data/stock_prices_raw.json data/stock_prices_final.json
```

### 2. 特征工程 (使用预训练好的word2vec词向量将数据集转变成向量表示)

```python
python src/gen_feature_matrix.py news_file final_prices_file output_feature_matrix word2vec_file sentense_len term_type news_type
python src/gen_feature_matrix.py data/news_reuters.csv data/stock_prices_final.json data/featureMatrix_ data/GoogleNews-vectors-negative300.bin 20 short headline
```
注意： 这里按照8:1:1的比例将数据集划分为训练集，验证集和测试集；同时还可以调整term_type（short middle long）获取短期、中期、长期的数据；也可以调整news_type（headline content）获取新闻标题、新闻内容的数据；如果需要重跑特征工程程序，请删除原有output_feature_matrix为前缀的文件，否则会在原有文件基础上进行写入

### 3. 模型训练
To train a Stacked-Bidirectional GRU network to predict the stock price movement.

```python
python model_sb_gru.py
```

### 4. Model
Model folder contains a trained model with 100K news on Stacked-Bidirectional GRU network.
To use the model

```python
cd model
python load.py
```
Accuracy of model is 96.79%. Furthur improvement like attention mechanism or cyclic learning rate is required. 

