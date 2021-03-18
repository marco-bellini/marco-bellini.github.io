---
  
title: "TW 1: Twitter Disaster Messages"
permalink: /twitter/initial_approach/
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




## 1. Introduction

The dataset is made available by [Figure Eight](https://www.figure-eight.com) and it comprises a set of messages related to disaster response, covering multiple languages, suitable for text categorization and related natural language processing tasks.


As described [here](https://www.figure-eight.com/dataset/combined-disaster-response-data/), this dataset contains 30,000 messages drawn from events including an earthquake in Haiti in 2010, an earthquake in Chile in 2010, floods in Pakistan in 2010, super-storm Sandy in the U.S.A. in 2012, and news articles spanning a large number of years and 100s of different disasters. The data has been encoded with 36 different categories related to disaster response and has been stripped of messages with sensitive information in their entirety.

Upon release, this is the featured dataset of a new Udacity course on Data Science and the AI4ALL summer school and is especially utile for text analytics and natural language processing (NLP) tasks and models.


### 2 Problem Assessment and Goal

The goal is to train a classifier able to detect different emergencies or disaster situations.

The problem is challenging because:
* disasters are infrequent events, hence the classes are imbalanced
* there are many misspellings in the messages and frequent use of foreign terms
* some emergency categories (e.g. `child alone`) are not represented in the training data
* some messages are very ambiguous (even to a human) and there is a small fraction of missclassified messages

In this case we limit ourselves to sketching a proof of concept, rather than presenting a full solution. 
For example, although the problem is multi-class we focus on predicting a single class (`food`).

We will evaluate the classification based on the `F1` metric and the confusion matrix (plotted as function of the threshold).
The goal is to have an acceptable number of detection of disaster events, keeping false negatives to a minimum.

We compare an approach based on word frequency (TFIDF) with an approach based on word embeddings and neural networks (through the [SpaCy](https://spacy.io/) library).


### 3 Exploratory data analysis

This is an excerpt of the data set:

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>message</th>
      <th>genre</th>
      <th>related</th>
      <th>PII</th>
      <th>request</th>
      <th>offer</th>
      <th>aid_related</th>
      <th>medical_help</th>
      <th>medical_products</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>Weather update - a cold front from Cuba that could pass over Haiti</td>
      <td>direct</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7</td>
      <td>Is the Hurricane over or is it not over</td>
      <td>direct</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>12</td>
      <td>says: west side of Haiti, rest of the country today and tonight</td>
      <td>direct</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>14</td>
      <td>Information about the National Palace-</td>
      <td>direct</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>15</td>
      <td>Storm at sacred heart of jesus</td>
      <td>direct</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




We note that several disaster situations are a very small fraction of the training set (large class imbalance):

{% raw %}![alt](/assets/nlp_disaster/class_imbalance.png){% endraw %}

---


We can have a look at some messages below:


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>message</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Weather update - a cold front from Cuba that could pass over Haiti</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Is the Hurricane over or is it not over</td>
    </tr>
    <tr>
      <th>2</th>
      <td>says: west side of Haiti, rest of the country today and tonight</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Information about the National Palace-</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Storm at sacred heart of jesus</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Please, we need tents and water. We are in Silo, Thank you!</td>
    </tr>
    <tr>
      <th>6</th>
      <td>I would like to receive the messages, thank you</td>
    </tr>
    <tr>
      <th>7</th>
      <td>There's nothing to eat and water, we starving and thirsty.</td>
    </tr>
    <tr>
      <th>8</th>
      <td>I am in Petionville. I need more information regarding 4636</td>
    </tr>
    <tr>
      <th>9</th>
      <td>I am in Thomassin number 32, in the area named Pyron. I would like to have some water. Thank God we are fine, but we desperately need water. Thanks</td>
    </tr>
  </tbody>
</table>
</div>


## 4 Baseline model with Bag-of-words 

The [Bag-of-words](https://en.wikipedia.org/wiki/Bag-of-words_model) (BOW) is a text classification approach that discards grammar and word order but focuses only on relative term frequency in a corpus of documents. 
First the corpus of messages is scanned and the mose representative words selected. Then the count of their occurrences in the messages is recorded in a sparse matrix that is processed by the [TF-IDF](https://en.wikipedia.org/wiki/Tf%E2%80%93idf) algorithm.

The basic idea is that a message about food will have a higher frequency of food related words than a message about storms.
So a classifier can be trained to recognize the topic of the messages depending on the relative frequency of the words in the message with respect to the corpus of messages used for training.


As a baseline we fit the simple model below:

```python
cv=CountVectorizer(tokenizer = tokenizer, max_df=max_df,min_df=min_df, max_features=max_features)
tfidf=TfidfTransformer()     
clf=RandomForestClassifier(  criterion="entropy", n_estimators=100)
```


|   | predicted 0  | predicted 1  |
|---|---|---|
|  true 0 | tn= 2294  | fp= 9  |
|  true 1 | fn= 163  |  tp= 163 |

This simple approach yields very low false positives but results in a recall of 0.5.
We can assume that several people will send out different messages in case of an emergency, but as we can observe from the analysis below with a low recall we would need large number of messages to miss only one disaster event in 100'000. 

{% raw %}![alt](/assets/nlp_disaster/multiple_events_detection.png){% endraw %}

Increasing recall by 50% without significant increase in the number of false positives would reduce the number of independent messages needed by half.


## 5 Improving the model

### 5.1 Preprocessing

Preprocessing is crucial: choosing the wrong words will significantly degrade the performance of the model.

The CountVectorizer tool is used to extract the most relevant (typically the most frequently used) words.

If we use the CountVectorizer function naively naively, we observe that:
* commond words with little predicted value (and, like, be, but ...) are included.
* names of countries, cities and nationalities may also included


```python
bow = CountVectorizer(max_features=1000,)
bow.fit(df.loc[0:1000,'message'])
print(bow.get_feature_names()[0:100])

```

    ['10', '11', '12', '12th', '15', '150', '16', '18', '19', '20', '200', '2000', '2010', '22', '23', '24', '27', '29', '30', '31', '32', '33', '35', '37', '3rd', '40', '41', '42', '43', '4636', '48', '4th', '50', '500', '52', '54', '600', '75', '79', '87', 'able', 'about', 'abroad', 'academy', 'access', 'account', 'across', 'action', 'actually', 'address', 'adopt', 'adress', 'advance', 'advertising', 'advise', 'after', 'afternoon', 'aftershake', 'aftershock', 'aftershocks', 'again', 'aid', 'aide', 'aids', 'airport', 'airtime', 'alerte', 'alexandre', 'alive', 'all', 'almost', 'alone', 'along', 'alot', 'already', 'also', 'always', 'am', 'ambroise', 'american', 'americans', 'an', 'and', 'another', 'anse', 'answer', 'antoine', 'any', 'anymore', 'anyone', 'anything', 'aquin', 'are', 'area', 'areas', 'aren', 'army', 'around', 'arrived', 'artibonit']


To prevent such issues and in the interest of efficiency, we use a more sophisticated tokenizer (based on SpaCy) that can:
* remove common English stop words
* remove Named Entities such as countries, cities, organizations etc

We can see an example of SpaCy Named entity recognition capabilities below:

```python
doc = nlp("I am planning to travel to Paris on Monday with George to visit the Louvre. Unfortunately I don't speak French well.")
displacy.render(doc, style="ent")
```


<div class="entities" style="line-height: 2.5; direction: ltr">I am planning to travel to 
<mark class="entity" style="background: #feca74; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em; box-decoration-break: clone; -webkit-box-decoration-break: clone">
    Paris
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">GPE</span>
</mark>
 on 
<mark class="entity" style="background: #bfe1d9; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em; box-decoration-break: clone; -webkit-box-decoration-break: clone">
    Monday
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">DATE</span>
</mark>
 with 
<mark class="entity" style="background: #aa9cfc; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em; box-decoration-break: clone; -webkit-box-decoration-break: clone">
    George
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">PERSON</span>
</mark>
 to visit the 
<mark class="entity" style="background: #feca74; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em; box-decoration-break: clone; -webkit-box-decoration-break: clone">
    Louvre
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">GPE</span>
</mark>
. Unfortunately I don't speak 
<mark class="entity" style="background: #c887fb; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em; box-decoration-break: clone; -webkit-box-decoration-break: clone">
    French
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">NORP</span>
</mark>
 well.</div>

We define a more sophisticated custom tokenizer:

```python
 
def spacy_tokenizer_stemmer(message):
    message=re.sub("[1-9#@$'!*+%\".()!,]?;",'',message).replace('','').replace('-','')
    message=' '.join(message.split())
    doc=nlp(message)
    words=[]

    stemmer = PorterStemmer()  
    
    remove_ent=[]
    for ent in doc.ents:
        if ent.label_ in ['GPE','LOC','NORP','FAC','ORG','LANGUAGE']:
            remove_ent.append(ent.text)

    # remove punctuation etc
    for token in doc:
        if ( (~token.is_stop)   & (token.pos_!='NUM') & (token.pos_!='PUNCT') & (token.pos_!='SYM') &
           ~(token.text in (remove_ent)) & (len(token.text)>1) ):
            words.append( stemmer.stem(token.text) )
    return(words)

```

The result is a better selection of words:

```python
bow = CountVectorizer(tokenizer = spacy_tokenizer_stemmer, max_df=0.9,min_df=3, max_features=1000)
bow.fit(df.loc[0:1000,'message'])

print(bow.get_feature_names()[0:50])
```

    ['12th', '3rd', '4th', 'able', 'abroad', 'access', 'address', 'adress', 'advance', 'afternoon', 'aftershake', 'aftershock', 'aftershocks', 'aid', 'aide', 'airport', 'alive', 'alot', 'answer', 'antoine', 'anymore', 'aquin', 'area', 'areas', 'arrived', 'artibonite', 'ask', 'asking', 'assistance', 'association', 'attention', 'au', 'aucun', 'authorities', 'available', 'avenue', 'away', 'b.', 'baby', 'bad', 'beach', 'bertin', 'besoin', 'better', 'big', 'bless', 'bodies', 'body', 'bois', 'bon']


We note that in the second case the words selected appear certainly more useful.

---


In general it is highly recommended to check the quality words choosen by the algorithm:
* for a small subset of messages to catch obvious issues (too many numbers, ...)
* on the positive disaster messages on a few classes to see if obvious trigger words are identified

We note that if the bag of words is too small we risk to miss important words (especially in the case of imbalanced classes).
Conversely if the bag of words is too big, the time needed to fit the model will increase.



### 5.2 Classification

After choosing the parameters for the CountVectorizer and TfidfTransformer algorithms we split the data into a test and training portion. Then the features obtained from the B.O.W. model are fed into a classifer.
We will compare the results of different classification algorithms.

The first choice is the size of the bag of words. Each word will be a feature for the classifier. With too few features, some critical words will not be included and no tuning of the classifier will correct this issue.
With too many words, the classifier will have to deal with many uninformative features.

```python
pipe1k=Pipeline([
    ('bow',CountVectorizer(tokenizer = spacy_tokenizer_stemmer, max_df=0.9,min_df=2, max_features=1000)),
    ('tfidf', TfidfTransformer() )    
])

```

The creation of the bag of word model and the calculation of frequencies is expensive. We start with a relatively small bag of words, resulting in 1000 features, corresponding to the words chosen. Because of the class imbalance 



We use the concept of learning curves to estimate which kind of classifier is best suited to this problem.


### 5.2.1 LogisticRegression Classifier

We start with a reasonable choice of classifier and we plot the learning curves with `roc_auc` and `F1` as scoring functions.
The LogisticRegression classifier is fast and the L1 regularization will push to zero the weights of the words that are not strongly related to the food class.


```python
clf=LogisticRegression(random_state=0, solver='liblinear',penalty='l1',max_iter=200,class_weight='balanced' )
 
dfh.plot_learning_curve(clf, 'LogisticRegression ROC AUC', bow1k_unb['Xtrain'], bow1k_unb['Ytrain'], ylim=None, cv=5, 
                    n_jobs=4, train_sizes=np.linspace(.1, 1.0, 5), scoring='roc_auc')
```


{% raw %}![alt](/assets/nlp_disaster/output_27_1.png){% endraw %}

The `roc_auc` does not seem very informative.



```python
clf=LogisticRegression(random_state=0, solver='liblinear',penalty='l1',max_iter=200,class_weight='balanced' )
dfh.plot_learning_curve(clf, 'LogisticRegression F1', bow1k_unb['Xtrain'], bow1k_unb['Ytrain'], ylim=None, cv=5, 
                    n_jobs=4, train_sizes=np.linspace(.1, 1.0, 5), scoring='f1');
```



{% raw %}![alt](/assets/nlp_disaster/output_28_1.png){% endraw %}


As shown above, if the we need a single figure to compare a model, the `F1` metric is preferable in case of unbalanced classes.

The learning curve is done by separating the original train dataset in a temporary train and test data in a cross-validation loop to average out the results over the CV folds. 
What is most important is that only a fraction (typically 10%, 30%, 50%... 100%) of the is used to fit the model. The test data is always used completely. This technique simulates how the model would behave if more data were made available. This is a different situation from a validation curve where the entirety of the training data is used and the complexity of the model in increased. In fact the complexity of the model is kept constant for the learning curves.

In this case, `F1` is decreasing in the training set. This means that *at fixed complexity* the model cannot generalize enough.  

The learning curve for training (in red) show that LogisticRegression does not improve enough on the training set for larger amounts of data. We need a classifier with more complexity (i.e. lower bias).
Therefore we try a RandomForestClassifier.

We also note that the gap between temporary training and test data is quite large. This is an indication that there are likely further problems.


### 5.2.2 RandomForest Classifier

We choose a RandomForest classifier because the complexity of the classifier can be increased with the number of estimators.


```python
N_est=5
clf= RandomForestClassifier(  criterion="entropy",class_weight="balanced", n_estimators=N_est)

dfh.plot_learning_curve(clf, 'RandomForest N=%d' % N_est, bow1k_unb['Xtrain'], bow1k_unb['Ytrain'], ylim=None, cv=5, 
                    n_jobs=4, train_sizes=np.linspace(.1, 1.0, 5), scoring='f1');
```


{% raw %}![alt](/assets/nlp_disaster/output_31_1.png){% endraw %}

On the other side a RandomForest classifier with a low number of trees (5) has a much flatter profile for the training set. We notice that it also has generalization problems with very large numbers of training examples because the number of trees is very low.


```python
N_est=50
clf= RandomForestClassifier(  criterion="entropy",class_weight="balanced", n_estimators=N_est)

dfh.plot_learning_curve(clf, 'RandomForest N=%d' % N_est, bow1k_unb['Xtrain'], bow1k_unb['Ytrain'], ylim=None, cv=5, 
                    n_jobs=4, train_sizes=np.linspace(.1, 1.0, 5), scoring='f1')
```

{% raw %}![alt](/assets/nlp_disaster/output_33_1.png){% endraw %}



We note that the RandomForest classifier with a higher number of tree has better behavior for the training set but does not improve in the test set (green).


```
    accuracy_score:  0.9528337771015595
    average precision:  0.8732778393558183
    balanced_accuracy_score:  0.8466817088406959
    ROC AUC:  0.9645514386409832
    
                  precision    recall  f1-score   support
    
               0       0.96      0.99      0.97      2303
               1       0.89      0.71      0.79       326
    
        accuracy                           0.95      2629
       macro avg       0.93      0.85      0.88      2629
    weighted avg       0.95      0.95      0.95      2629
    


    Confusion Matrix
    C true,predicted
    
    [[2278   25]
     [  94  232]]
    
    true negatives  : true 0, predicted 0:  2278
    false positives : true 0, predicted 1:  25
    false negatives : true 1, predicted 0:  94
    true positives  : true 1, predicted 1:  232

```


Considering the challenging problem (NLP, imbalanced classes) we note that even before optimizing the hyperparameters the results look acceptable.

However, before going further it is important to try to understand the reason for the false negatives. 
Does the B.O.W. contain the right words?


```python
X_test=test_df['message']
y_fn=((Y_test==1) & (y_pred==0))
 
display(X_test.loc[y_fn].head(20).to_frame())
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>message</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>18</th>
      <td>they are in a area near a mountain hungry, sick their is no water please remember us</td>
    </tr>
    <tr>
      <th>49</th>
      <td>we do not have water, our house is compromised (cracked), we do not have a place where to sleep.</td>
    </tr>
    <tr>
      <th>80</th>
      <td>I need to let my family know that I am starving, they need to send me money. What Money transfer company is openin my area..</td>
    </tr>
    <tr>
      <th>167</th>
      <td>URGENT, please help us, children and olders are dying from starvation. Please you have to come, we are downtown Leogane, rue ( street ) Danjour. Thank you</td>
    </tr>
    <tr>
      <th>169</th>
      <td>SOS, SOS. 27 month old child in difficulties at Cap-haitien. We need diapers, milk and protein. Thank you for your help.</td>
    </tr>
    <tr>
      <th>256</th>
      <td>People in Thomazeau don't have anything to eat, especially in Hatte Cadete</td>
    </tr>
    <tr>
      <th>266</th>
      <td>I live in martisan and i want to know where i can find clean water today because since thusday i havent found water. I'm starving, please help me</td>
    </tr>
    <tr>
      <th>273</th>
      <td>My child is dying of starvation, I have received nothing</td>
    </tr>
    <tr>
      <th>286</th>
      <td>No Location : People are starving. .. please do something to save us.</td>
    </tr>
    <tr>
      <th>346</th>
      <td>I am in Turtle (Tortuga) Island, I did not find help and I am starving..</td>
    </tr>
    <tr>
      <th>351</th>
      <td>IN ST March, people are starving and are sick,we are waiting for help..</td>
    </tr>
    <tr>
      <th>398</th>
      <td>my kids died, my house is destroyed. i'm beating..</td>
    </tr>
    <tr>
      <th>415</th>
      <td>this person wants to know if only the people with card can get food?</td>
    </tr>
    <tr>
      <th>471</th>
      <td>How do I find coupons I can use to go get food given by PAM (World Food Program)?</td>
    </tr>
    <tr>
      <th>480</th>
      <td>There are some victims at my house, we have eaten all we have.</td>
    </tr>
    <tr>
      <th>484</th>
      <td>there is a one month old baby and the doctor told the mother not to nurse because the milk is bad. Please what should we do. We need milk for the baby.</td>
    </tr>
    <tr>
      <th>533</th>
      <td>WHAT CAN I DO WHEN THE FOOD IS FINISHED RIGHT IN FRONT OF ME,WHAT CAN I DO WITH THE CARD?</td>
    </tr>
    <tr>
      <th>550</th>
      <td>Hello, I'd like to know what number to call for the World Food Program, to thank them.</td>
    </tr>
    <tr>
      <th>809</th>
      <td>We need tent, cover, rice. Uneted Nation never Help us since the earthquake, we live in Carre-four, Lapot street, We want to know if we are not victims, cause we never recieved any visit from the leaders to let us know what is about.</td>
    </tr>
    <tr>
      <th>821</th>
      <td>me ever to find the rices that were distribute by the American. makes me find rices</td>
    </tr>
  </tbody>
</table>
</div>


We use `eli5` to understand why some predictions are off.

```python
full_pipe=Pipeline([
    ('bow', pipe1k.steps[0][1]),
    ('tfidf', pipe1k.steps[1][1]),    
    ('clf', clf),    
])

explain_message(80,X_test[y_fn],full_pipe,bow1k_unb)


    original:  I need to let my family know that I am starving, they need to send me money. What Money transfer company is openin my area..
    transformed:  area,compani,famili,know,let,money,need,send
    
    Predicted class: 0

```

<div>
    <style>
    table.eli5-weights tr:hover {
        filter: brightness(85%);
    }
</style>



    

    

    

    

    

    


    

    

    

    
        

    

        
            
                
                
    
        <p style="margin-bottom: 0.5em; margin-top: 0em">
            <b>
    
        y=0
    
</b>

    
    (probability <b>0.940</b>)

top features
        </p>
    
    <table class="eli5-weights"
           style="border-collapse: collapse; border: none; margin-top: 0em; table-layout: auto; margin-bottom: 2em;">
        <thead>
        <tr style="border: none;">
            
                <th style="padding: 0 1em 0 0.5em; text-align: right; border: none;" title="Feature contribution already accounts for the feature value (for linear models, contribution = weight * feature value), and the sum of feature contributions is equal to the score or, for some classifiers, to the probability. Feature values are shown if &quot;show_feature_values&quot; is True.">
                    Contribution<sup>?</sup>
                </th>
            
            <th style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">Feature</th>
            
        </tr>
        </thead>
        <tbody>
        
            <tr style="background-color: hsl(120, 100.00%, 80.00%); border: none;">
    <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
        +0.501
    </td>
    <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
        &lt;BIAS&gt;
    </td>
    
</tr>
        
            <tr style="background-color: hsl(120, 100.00%, 86.36%); border: none;">
    <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
        +0.290
    </td>
    <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
        food
    </td>
    
</tr>
        
            <tr style="background-color: hsl(120, 100.00%, 95.31%); border: none;">
    <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
        +0.063
    </td>
    <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
        water
    </td>
    
</tr>
        
            <tr style="background-color: hsl(120, 100.00%, 97.06%); border: none;">
    <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
        +0.032
    </td>
    <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
        know
    </td>
    
</tr>
        
            <tr style="background-color: hsl(120, 100.00%, 97.22%); border: none;">
    <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
        +0.030
    </td>
    <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
        famili
    </td>
    
</tr>
        
            <tr style="background-color: hsl(120, 100.00%, 97.87%); border: none;">
    <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
        +0.020
    </td>
    <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
        hungri
    </td>
    
</tr>
        
            <tr style="background-color: hsl(120, 100.00%, 98.21%); border: none;">
    <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
        +0.016
    </td>
    <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
        eat
    </td>
    
</tr>
        
        
            <tr style="background-color: hsl(120, 100.00%, 98.21%); border: none;">
                <td colspan="2" style="padding: 0 0.5em 0 0.5em; text-align: center; border: none; white-space: nowrap;">
                    <i>&hellip; 408 more positive &hellip;</i>
                </td>
            </tr>
        

        
            <tr style="background-color: hsl(0, 100.00%, 97.71%); border: none;">
                <td colspan="2" style="padding: 0 0.5em 0 0.5em; text-align: center; border: none; white-space: nowrap;">
                    <i>&hellip; 176 more negative &hellip;</i>
                </td>
            </tr>
        
        
            <tr style="background-color: hsl(0, 100.00%, 97.71%); border: none;">
    <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
        -0.023
    </td>
    <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
        money
    </td>
    
</tr>
        
            <tr style="background-color: hsl(0, 100.00%, 97.57%); border: none;">
    <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
        -0.025
    </td>
    <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
        let
    </td>
    
</tr>
        
            <tr style="background-color: hsl(0, 100.00%, 91.06%); border: none;">
    <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
        -0.159
    </td>
    <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
        need
    </td>
    
</tr>
        

        </tbody>
    </table>

            
        

        



    

    

    

    


    

    

    

    

    

    

</div>
    

    

    

    

    

    


It seems that the B.O.W. Does not contain the word "starving".


```python
words=pipe1k.steps[0][1].get_feature_names()

check_word='starving'
display( 'is %s in B.O.W. ? %s' %  (check_word, check_word in words))

n_check_word=df.loc[df['message'].str.contains('starv'),'message'].shape[0]
print('training set messages with %s: %s' % (check_word,n_check_word) )


    'is starving in B.O.W. ? False'


    training set messages with starving: 55

```


```python
explain_message(415,X_test[y_fn],full_pipe,bow1k_unb)


    original:  this person wants to know if only the people with card can get food? 
    transformed:  card,food,know,peopl,person,want
    
    Predicted class: 0

```

<div>
    <style>
    table.eli5-weights tr:hover {
        filter: brightness(85%);
    }
</style>



    

    

    

    

    

    


    

    

    

    
        

    

        
            
                
                
    
        <p style="margin-bottom: 0.5em; margin-top: 0em">
            <b>
    
        y=0
    
</b>

    
    (probability <b>0.540</b>)

top features
        </p>
    
    <table class="eli5-weights"
           style="border-collapse: collapse; border: none; margin-top: 0em; table-layout: auto; margin-bottom: 2em;">
        <thead>
        <tr style="border: none;">
            
                <th style="padding: 0 1em 0 0.5em; text-align: right; border: none;" title="Feature contribution already accounts for the feature value (for linear models, contribution = weight * feature value), and the sum of feature contributions is equal to the score or, for some classifiers, to the probability. Feature values are shown if &quot;show_feature_values&quot; is True.">
                    Contribution<sup>?</sup>
                </th>
            
            <th style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">Feature</th>
            
        </tr>
        </thead>
        <tbody>
        
            <tr style="background-color: hsl(120, 100.00%, 80.00%); border: none;">
    <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
        +0.501
    </td>
    <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
        &lt;BIAS&gt;
    </td>
    
</tr>
        
            <tr style="background-color: hsl(120, 100.00%, 94.09%); border: none;">
    <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
        +0.088
    </td>
    <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
        know
    </td>
    
</tr>
        
            <tr style="background-color: hsl(120, 100.00%, 95.99%); border: none;">
    <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
        +0.050
    </td>
    <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
        want
    </td>
    
</tr>
        
            <tr style="background-color: hsl(120, 100.00%, 96.32%); border: none;">
    <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
        +0.045
    </td>
    <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
        card
    </td>
    
</tr>
        
            <tr style="background-color: hsl(120, 100.00%, 97.27%); border: none;">
    <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
        +0.029
    </td>
    <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
        avail
    </td>
    
</tr>
        
            <tr style="background-color: hsl(120, 100.00%, 97.29%); border: none;">
    <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
        +0.029
    </td>
    <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
        need
    </td>
    
</tr>
        
            <tr style="background-color: hsl(120, 100.00%, 97.47%); border: none;">
    <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
        +0.026
    </td>
    <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
        person
    </td>
    
</tr>
        
            <tr style="background-color: hsl(120, 100.00%, 97.72%); border: none;">
    <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
        +0.022
    </td>
    <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
        water
    </td>
    
</tr>
        
        
            <tr style="background-color: hsl(120, 100.00%, 97.72%); border: none;">
                <td colspan="2" style="padding: 0 0.5em 0 0.5em; text-align: center; border: none; white-space: nowrap;">
                    <i>&hellip; 214 more positive &hellip;</i>
                </td>
            </tr>
        

        
            <tr style="background-color: hsl(0, 100.00%, 95.22%); border: none;">
                <td colspan="2" style="padding: 0 0.5em 0 0.5em; text-align: center; border: none; white-space: nowrap;">
                    <i>&hellip; 413 more negative &hellip;</i>
                </td>
            </tr>
        
        
            <tr style="background-color: hsl(0, 100.00%, 95.22%); border: none;">
    <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
        -0.065
    </td>
    <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
        peopl
    </td>
    
</tr>
        
            <tr style="background-color: hsl(0, 100.00%, 85.04%); border: none;">
    <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
        -0.331
    </td>
    <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
        food
    </td>
    
</tr>
        

        </tbody>
    </table>

            
        

        



    

    

    

    


    

    

    

    

    

    


    

    

    

    

    

    
</div>





### 5.3 Discussion of model choice


First of all we notice that using an advanced preprocessor has significantly reduced the number of false negatives.


Then, looking at the false negatives and at the original messages we find two separate issues:

The first issue is that the B.O.W. misses critical words.
Although several examples were provided in the training set of the word "starving", it was not included in the B.O.W. 
With such poor feature selection, there is nothing that the classifier can do afterwards.

We have two approaches to correct this issue:
* increase the number of features in the B.O.W.
* rebalance the training set by undersampling the Y=0 messages

Looking at the learning curves, we conclude that the first approach looks feasible.
We will also add a stemmer to catch all the occurances of (starving, starved, starvation) with a single feature.


The second issue in the case of a RandomForest classifier is that the B.O.W. contains the relevant word`food` but unfortunately the many other features (that are not very predictive) take over. 
In other words the many small weights of the unrelated words overcome the weight of the single relevant word.

This example suggests that introducing stronger regularization would be beneficial, so that the large number of non-predictive features has weights pulled towards zero.



