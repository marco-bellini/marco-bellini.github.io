---
  
title: "TW 2: Bag of Word approach"
permalink: /twitter/bow_refinement/
excerpt: "Twitter Response: Initial Bag-of-Words approach"
last_modified_at: 2019-04-18T15:53:52-04:00
toc: true
toc_label: "Page Index"
toc_sticky: true
author_profile: true
classes: post
tags:
  - Classification
categories:
  - Twitter
---

## 6 Feature Selection

Unsurprisingly the main issue in the problem under discussion is feature selection.
Only a few words in each messge are really critical. A BOW model with 5000 words will likely include 200 predictive words (i.e. features) and 4800 not predictive features.

This can be explained as follows:
* non disaster related messages are much more frequent, so it will be harder to pick the predictive words because of their relative low frequence of occurrence
* misspellings lower further the frequency of predictive words
* there are predictive words with positive predictive value (i.e. `food` is a strong predictor of a food related emergency, especially combined with `need` or `lack`) but it's difficult to find words that predict the absence of an emergency  

Therefore we try out two possible solutions:
* undersample the messages from the negative class to increase the probability to obtain more predictive words associated to food
* apply a feature selection technique based on `chi2` to select a smaller number of features with the highest values for the chi-squared statistic to remove the features that are independent of class.


### 6.1 Undersampling the negative messages

We define a function that keeps the full positive class examples but selects a sample of negative class examples. 
The goal is to force the B.O.W. to choose more appropriate words.

```python


def imbalanced_undersample(X_train,Y_train,n_false_sample, classes=[False,True]):
    """
    resamples the false class classes[0] with n_false_sample to correct for imbalanced data
    
    """
    
    n_false=Y_train.loc[Y_train==classes[0]].shape[0]
    n_true=Y_train.loc[Y_train==classes[1]].shape[0]
    ind_false=Y_train.loc[Y_train==classes[0]].index
    ind_true=Y_train.loc[Y_train==classes[1]].index

    print('Original n_true, n_false:' ,n_true,n_false)
    
    ind_s=sample_without_replacement(n_false,n_false_sample)
    ind=np.hstack((ind_true,ind_false[ind_s]))
    np.random.shuffle(ind)

    X_train2=X_train.iloc[ind].copy()
    Y_train2=Y_train.iloc[ind].copy()
    
    n_false2=Y_train2.loc[Y_train==classes[0]].shape[0]
    n_true2=Y_train2.loc[Y_train==classes[1]].shape[0]
    
    print('Resampled n_true, n_false:' ,n_true2,n_false2)
    
    return(X_train2,Y_train2)
```


```python
n_false_sample=2400
X_train2,Y_train2=imbalanced_undersample(X_train,Y_train,n_false_sample, classes=[False,True])

Y_train2.hist()
```


{% raw %}![alt](/assets/nlp_disaster/output_48_1.png){% endraw %}


The new B.O.W. contains the word stem `starv` for starving.

```python
bow_bal=CountVectorizer(tokenizer = spacy_tokenizer_stemmer, max_df=0.9,min_df=1, max_features=2000)
words=bow_bal.get_feature_names()

check_word='starv'
display( 'is %s in B.O.W. ? %s' %  (check_word, check_word in words))
```


    'is starv in B.O.W. ? True'

However, there's no reason to discard the negative messages once the vectorizer is trained in this way we choose better the words of the B.O.W. but we retain the number of messages


```python

Xbow_train=bow_bal.transform(X_train)
X_train_tdidf2 = tfidf.fit_transform(Xbow_train)

Xbow_test = bow_bal.transform(X_test)
X_test_tdidf2 = tfidf.transform(Xbow_test)

bow1k_bal={'Xtrain':X_train_tdidf2, 'Ytrain':Y_train, 'Xtest':X_test_tdidf2, 'Ytest':Y_test}


```



### 6.2 Feature Selection


```python

fs=SelectKBest(chi2, k=k) 

```


### 6.3 Discussion of Results


We compare two BOW with 100 words:
* BOW1 is derived roughly the same number of positive and negative words
* BOW2 is derived keeping the 100 most significative words from a BOW of 5000 words obtained from the whole corpus of messages




