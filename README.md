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
python src/crawler_tickers.py 20 ../data/ticker_list.csv
```
注意: 20表示市场容量为前20%，可以调整；    
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;这里要用python2 urllib2抓数据，因为尝试用python3 urllib3抓数据时会有网络错误

#### 1.2 从Reuters抓取股票代码相关的财经新闻

```python
python src/crawler_reuters.py ../data/ticker_list.csv ../data/finished_tickers.csv ../data/news_failed_tickers.csv ../data/news_reuters.csv  
```
注意: 第一次抓新闻的时候，要确保../data/finished_tickers.csv文件为空，否则会过滤相关股票的新闻

#### 1.3 从Yahoo Finance抓取股票代码相关的价格信息

```python
python src/crawler_yahoo_finance.py 
```

#### 1.4 对抓取的价格信息进行处理，获取每只股票短期、中期、长期的价格变动

```python
python src/create_label.py 
```

### 2. Feature Engineering (Convert word to word vector based on pretrained glove vector)

Projection of word to word vector
Seperate test set away from training+validation test, otherwise we would get a too optimistic result.

```python
python genFeatureMat_glove.py
```

### 3. Training 
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

