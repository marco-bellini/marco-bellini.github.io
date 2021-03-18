---
  
title: "AR 1: Responses to Mail Order Campaign"
permalink: /arvato/initial_approach/
excerpt: "Arvato Customer Response: Initial approach"
last_modified_at: 2019-04-18T15:53:52-04:00
toc: true
toc_label: "Page Index"
toc_sticky: true
author_profile: true
classes: post
tags:
  - Classification
categories:
  - Arvato
---




## 1. Introduction

The dataset is provided by Arvato Financial Solutions, a Bertelsmann subsidiary and it comprises demographic attributes from the targets of a mailing order campaign. The task is to build a machine learning model that predicts whether or not each individual will respond to the campaign.

### 2 Problem Assessment and Goal

The goal is to train a classifier able to  .

The problem is challenging because:
* disasters are infrequent events, hence the classes are imbalanced

In this case we limit ourselves to sketching a proof of concept, rather than presenting a full solution. 
For example, although the problem is multi-class we focus on predicting a single class (`food`).

We will evaluate the classification based on the `F1` metric and the confusion matrix (plotted as function of the threshold).
The goal is to have an acceptable number of detection of disaster events, keeping false negatives to a minimum.


### EDA

The dataset comprises 360 features. We perform minor data cleaning operations (e.g. many values are used for nan, such as "-1", "X", "XX" etc ).
After binning a small number of features we are left with a dataset comprising only ordered categorical variables.




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
      <th>AGER_TYP</th>
      <th>AKT_DAT_KL</th>
      <th>ALTER_HH</th>
      <th>ALTERSKATEGORIE_FEIN</th>
      <th>ANZ_KINDER</th>
      <th>ARBEIT</th>
      <th>BALLRAUM</th>
      <th>CAMEO_DEU_2015</th>
      <th>CAMEO_DEUG_2015</th>
      <th>CAMEO_INTL_2015</th>
      <th>...</th>
      <th>WEALTH</th>
      <th>LIFESTAGE</th>
      <th>ANZ_HAUSHALTE_AKTIV_bin</th>
      <th>ANZ_HH_TITEL_bin</th>
      <th>ANZ_PERSONEN_bin</th>
      <th>ANZ_TITEL_bin</th>
      <th>ANZ_STATISTISCHE_HAUSHALTE_bin</th>
      <th>VERDICHTUNGSRAUM_bin</th>
      <th>MIN_GEBAEUDEJAHR_bin</th>
      <th>GEBURTSJAHR_bin</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>38282</th>
      <td>2.0</td>
      <td>1.0</td>
      <td>9.0</td>
      <td>9.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>6.0</td>
      <td>20.0</td>
      <td>5.0</td>
      <td>33.0</td>
      <td>...</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1994.0</td>
      <td>1950.0</td>
    </tr>
    <tr>
      <th>17413</th>
      <td>3.0</td>
      <td>1.0</td>
      <td>12.0</td>
      <td>12.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>8.0</td>
      <td>2.0</td>
      <td>14.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>6.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>6.0</td>
      <td>7.0</td>
      <td>1992.0</td>
      <td>1950.0</td>
    </tr>
    <tr>
      <th>15542</th>
      <td>3.0</td>
      <td>3.0</td>
      <td>18.0</td>
      <td>13.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>6.0</td>
      <td>21.0</td>
      <td>5.0</td>
      <td>34.0</td>
      <td>...</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>10.0</td>
      <td>1992.0</td>
      <td>1958.0</td>
    </tr>
    <tr>
      <th>35461</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>15.0</td>
      <td>2.0</td>
      <td>7.0</td>
      <td>2.0</td>
      <td>20.0</td>
      <td>10.0</td>
      <td>1998.0</td>
      <td>2005.0</td>
    </tr>
    <tr>
      <th>30825</th>
      <td>2.0</td>
      <td>1.0</td>
      <td>9.0</td>
      <td>11.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>2.0</td>
      <td>37.0</td>
      <td>8.0</td>
      <td>54.0</td>
      <td>...</td>
      <td>5.0</td>
      <td>4.0</td>
      <td>11.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>10.0</td>
      <td>1.0</td>
      <td>1992.0</td>
      <td>1950.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 360 columns</p>
</div>

We split the data in 70% training and 30% validation.
We also note that the dataset is extremely unbalanced.

```python
(100*y_train.value_counts(normalize=True)).plot(kind='bar')
plt.ylabel('percentage')
plt.xlabel('response')
plt.title('classes are extremely imbalanced')

```
 
 {% raw %}![alt](/assets/arvato/output_4_1.png){% endraw %}


## Feature selection

Since the features are categorical we can easily test which features are most important to predict the labels.
We can construct the contingency table for feature `D19_SOZIALES` 


```python
contingency_table=X.groupby(by='Y')['D19_SOZIALES'].value_counts().unstack()
display(contingency_table)
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
      <th>D19_SOZIALES</th>
      <th>0.0</th>
      <th>1.0</th>
      <th>2.0</th>
      <th>3.0</th>
      <th>4.0</th>
      <th>5.0</th>
    </tr>
    <tr>
      <th>Y</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6757</td>
      <td>7450</td>
      <td>1017</td>
      <td>6044</td>
      <td>2244</td>
      <td>984</td>
    </tr>
    <tr>
      <th>1</th>
      <td>21</td>
      <td>259</td>
      <td>7</td>
      <td>15</td>
      <td>8</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>

We notice that value 1 is disproportionally high in class 1 compared to class 0, as also show below with percentages of values by class: 


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
      <th>D19_SOZIALES</th>
      <th>0.0</th>
      <th>1.0</th>
      <th>2.0</th>
      <th>3.0</th>
      <th>4.0</th>
      <th>5.0</th>
    </tr>
    <tr>
      <th>Y</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>27.6</td>
      <td>30.4</td>
      <td>4.2</td>
      <td>24.7</td>
      <td>9.2</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>6.7</td>
      <td>83.0</td>
      <td>2.2</td>
      <td>4.8</td>
      <td>2.6</td>
      <td>0.6</td>
    </tr>
  </tbody>
</table>
</div>


Using statsmodels chi2_contingency test we can find the most significant features.
We show below p-values of chi2 test and the percentage of missing values:


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
      <th>p-value</th>
      <th>perc.missing</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>D19_SOZIALES</th>
      <td>3.938247e-84</td>
      <td>17.507399</td>
    </tr>
    <tr>
      <th>D19_KONSUMTYP_MAX</th>
      <td>2.380309e-34</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>D19_KONSUMTYP</th>
      <td>8.840219e-14</td>
      <td>17.507399</td>
    </tr>
    <tr>
      <th>RT_SCHNAEPPCHEN</th>
      <td>2.224525e-08</td>
      <td>1.443155</td>
    </tr>
    <tr>
      <th>KBA05_CCM4</th>
      <td>3.905764e-05</td>
      <td>20.902471</td>
    </tr>
    <tr>
      <th>KBA05_KW3</th>
      <td>7.780143e-05</td>
      <td>20.902471</td>
    </tr>
    <tr>
      <th>KBA05_ZUL4</th>
      <td>2.908871e-03</td>
      <td>20.902471</td>
    </tr>
    <tr>
      <th>KBA13_MERCEDES</th>
      <td>4.589483e-03</td>
      <td>18.431816</td>
    </tr>
    <tr>
      <th>CJT_KATALOGNUTZER</th>
      <td>5.520500e-03</td>
      <td>1.443155</td>
    </tr>
    <tr>
      <th>KBA05_KW2</th>
      <td>5.602754e-03</td>
      <td>20.902471</td>
    </tr>
  </tbody>
</table>
</div>


We can quickly verify our finding by fitting a simple DecisionTree model with minimal depth and using only the `D19_SOZIALES` feature.
Looking at the classification report and confusion matrix, we notice that this baseline model is not so bad as it can detect 88% of the positive responses with just a single feature.

However, as expected, given the class imbalance, the model also shows a large number of false positives. Our goal will be to try to reduce the number of false negative and positives.


```python
clfd = DecisionTreeClassifier(max_depth=2,class_weight='balanced')
X_train_dt1=X_train_imp['D19_SOZIALES'].values.reshape(-1, 1)

print(classification_report(y_test, y_pred))


              precision    recall  f1-score   support

           0       1.00      0.57      0.73     12729
           1       0.02      0.88      0.05       160

    accuracy                           0.57     12889
   macro avg       0.51      0.72      0.39     12889
weighted avg       0.99      0.57      0.72     12889

Confusion Matrix
C true,predicted

[[7268 5461]
 [  20  140]]

true negatives  : true 0, predicted 0:  7268
false positives : true 0, predicted 1:  5461
false negatives : true 1, predicted 0:  20
true positives  : true 1, predicted 1:  140
```

We can compare our baseline model with the results from XGBoost (a state-of-the-art classifier capable to handle nan) on the full dataset.

```python

xgb_model = xgb.XGBClassifier(silent=True,objective='binary:logistic',learning_rate= 0.01,
                                 scale_pos_weight=100.)

```

We run a three-fold stratified cross-validation comparing the two classifiers and choosing average precision as a metric:

 {% raw %}![alt](/assets/arvato/rule.png){% endraw %}

Again, we notice that the performance of the baseline classifier is quite reasonable, confirming that the single feature chosen is highly predictive.


The next step is to choose a how many features to retain from the 360 of the dataset.
Instead of the contingency tables, we can use the SelectKBest(chi2) function from sklearn to simplify the analysis, with a small loss in accuracy.

```python

pipe=Pipeline(steps=[
       ('fs',SelectKBest(chi2) ),
       ('clf',xgb.XGBClassifier(silent=True,objective='binary:logistic',learning_rate= 0.01,
                                 scale_pos_weight=100) )
])

param_range=[1,2,3,4,5,7,9,10,12,15,20,30,50,100,200,360]
# param_range=[1,3,9,20]

# scoring="roc_auc" #1
scoring="average_precision" 
skf = StratifiedKFold(n_splits=3)

train_scores, valid_scores = validation_curve(pipe, X_train_imp, y_train, "fs__k",
                    param_range,scoring=scoring,
                    cv=skf)
```


```python
ax=dfh.plot_validation_curve(train_scores,valid_scores,"n. features",'average precision')
plt.xscale('Log')
```

 {% raw %}![alt](/assets/arvato/output_12_0.png){% endraw %}

We notice that after 7-8 features there is neglible improvement in average precision.


```python
param_range=[9,30,50,100,150,200]
ax=dfh.plot_validation_curve(train_scores,valid_scores,"n. features",'average precision',title='Validation Curve cv=5',leg_loc=2)
plt.xscale('Log')
```

 {% raw %}![alt](/assets/arvato/output_15_0.png){% endraw %}







 
 