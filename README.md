# Headline-Driven Stock Prediction using Deep-Learning

It is useful to predict stock movement based on headline of news. In this project, we train a deep model for headline driven stock market prediction.

## Requirements

`Python 3`  
`pip install bs4`  

## Running Steps

1. Collection of Data ( essential and tricky task ) 

    1.1 get the whole ticker list

    1.2 crawl news 
    
    1.3 crawl prices using urllib2 (Yahoo Finance API is outdated)

2. Train the GloVe in corpus in NLTK

    2.1 build the word-word co-occurrence matrix
  
    2.2 factorizing the weighted log of the co-occurrence matrix
  
3. Feature Engineering
  
    3.2 Unify word format: unify tense, singular & plural, remove punctuations & stop words
  
    3.2 Extract feature using feature hashing based on the trained word vector (step 2)
  
    3.3 Pad word senquence (essentially a matrix) to keep the same dimension
  
4. Trained a ConvNet to predict the stock price movement based on a reasonable parameter selection
5. The result shows a significant 1-2% improve on the test set

Use the following script to crawl it and format it to our local file
Note : We can relate the news with company and date, this is more precise than Bloomberg News
### 1 Data Preparation

#### 1.1 crawl tickers (companies) list from NASDAQ 

```python
python src/crawler_allTickers.py 100 # select top 100% company list, the num is an ajustable parameter
```
Note:here need python2 urllib2, since python3 urllib3 meet http error here

#### 1.2 crawl headline of company news from reuters

```python
python src/crawler_reuters.py 
```

#### 1.3 crawl stock prices of companies from yahoo finance

```python
python src/crawler_yahoo_finance.py 
```

#### 1.4 align headline with stock movement and create label

```python
python src/create_label.py 
```

### 2 Word Embedding
To use our customized word vector, apply GloVe to train word vector from Reuters corpus in NLTK

```python
python embeddingWord.py
```
To use pre-trained GloVe word vectors

```python
python embeddingWordPre.py
```


### 3. Feature Engineering

Projection of word to word vector
Seperate test set away from training+validation test, otherwise we would get a too optimistic result.

```python
python genFeatureMatrix.py
```
Here there is important point to note when we are separating the Cross Validation set and the Training Set. The shuffiling of data can create a very large mistake and untraceble. Consider we have news that are similar in contex but the language of news are slightly different and the got separated in training and the cross validation set. Then the error in cross validation set get biased as the model is already trained against that model so that example will be of no use and effective cross validation set reduces.

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

