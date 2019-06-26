# Headline-Driven Stock Prediction using Deep-Learning

It is useful to predict stock movement based on headline of news. In this project, we train a deep model for headline driven stock market prediction.

## Requirements

`python3 # crawl ticker list need python2`  
`pip install bs4` 
`pip install gensim`  
`pip install theano`  
`pip install yfinance`   
`pip install keras`  


## Running Steps

### 1 Data Preparation

#### 1.1 crawl tickers (companies) list from NASDAQ 

```python
python src/crawler_allTickers.py 20 # select top 100% company list, the num is an ajustable parameter
```
Note:here need python2 urllib2, since python3 urllib3 meet http error here

#### 1.2 crawl headline of company news from reuters

```python
python src/crawler_reuters.py 
```

#### 1.3 crawl stock's kinds of prices of companies from yahoo finance

```python
python src/crawler_yahoo_finance.py 
```

#### 1.4 get price movement of each stock of each day

```python
python src/create_label.py 
```

### 2 Apply GloVe to train word vector from Reuters corpus in NLTK

```python
python embeddingWord.py
```

### 3. Feature Engineering (Convert word to word vector based on pretrained glove vector)

Projection of word to word vector
Seperate test set away from training+validation test, otherwise we would get a too optimistic result.

```python
python genFeatureMat_glove.py
```

### 4. Training 
To train a Stacked-Bidirectional GRU network to predict the stock price movement.

```python
python model_sb_gru.py
```

To train a ConvoNet network to predict the stock price movement.

```python
python model_cnn.py
```
### 5. Model
Model folder contains a trained model with 100K news on Stacked-Bidirectional GRU network.
To use the model

```python
cd model
python load.py
```
Accuracy of model is 96.79%. Furthur improvement like attention mechanism or cyclic learning rate is required. 

