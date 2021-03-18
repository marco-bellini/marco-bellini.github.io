---
title: "CCF 1: Model"
permalink: /credit_card_fraud/ccf_model/
excerpt: "Credit Card Fraud: Model"
last_modified_at: 2020-08-11T15:53:52-04:00
toc: true
toc_label: "Page Index"
toc_sticky: true
author_profile: false
classes: post
tags:
  - classification
  - EDA
  - unbalanced dataset
  - XGBoost
  - scikit-learn
  - learning curve
  - feature selection  
categories:
  - Credit Card Fraud
---

# ULB Credit Card Approval Data Analysis
 
The database under investigation is available at Kaggle: https://www.kaggle.com/mlg-ulb/creditcardfraud

The goal is to train a classifier able to distinguish fraudolent transactions (obviously a very small percentage of total transactions).


## Content
The datasets contains transactions made by credit cards in September 2013 by european cardholders.
This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions.

It contains only numerical input variables which are the result of a PCA transformation. Unfortunately, due to confidentiality issues, we cannot provide the original features and more background information about the data. Features V1, V2, … V28 are the principal components obtained with PCA, the only features which have not been transformed with PCA are 'Time' and 'Amount'. Feature 'Time' contains the seconds elapsed between each transaction and the first transaction in the dataset. The feature 'Amount' is the transaction Amount, this feature can be used for example-dependant cost-senstive learning. Feature 'Class' is the response variable and it takes value 1 in case of fraud and 0 otherwise.

## Acknowledgements
The dataset has been collected and analysed during a research collaboration of Worldline and the Machine Learning Group (http://mlg.ulb.ac.be) of ULB (Université Libre de Bruxelles) on big data mining and fraud detection.
More details on current and past projects on related topics are available on https://www.researchgate.net/project/Fraud-detection-5 and the page of the DefeatFraud project

### References:

Andrea Dal Pozzolo, Olivier Caelen, Reid A. Johnson and Gianluca Bontempi. Calibrating Probability with Undersampling for Unbalanced Classification. In Symposium on Computational Intelligence and Data Mining (CIDM), IEEE, 2015

Dal Pozzolo, Andrea; Caelen, Olivier; Le Borgne, Yann-Ael; Waterschoot, Serge; Bontempi, Gianluca. Learned lessons in credit card fraud detection from a practitioner perspective, Expert systems with applications,41,10,4915-4928,2014, Pergamon

Dal Pozzolo, Andrea; Boracchi, Giacomo; Caelen, Olivier; Alippi, Cesare; Bontempi, Gianluca. Credit card fraud detection: a realistic modeling and a novel learning strategy, IEEE transactions on neural networks and learning systems,29,8,3784-3797,2018,IEEE

Dal Pozzolo, Andrea Adaptive Machine learning for credit card fraud detection ULB MLG PhD thesis (supervised by G. Bontempi)

Carcillo, Fabrizio; Dal Pozzolo, Andrea; Le Borgne, Yann-Aël; Caelen, Olivier; Mazzer, Yannis; Bontempi, Gianluca. Scarff: a scalable framework for streaming credit card fraud detection with Spark, Information fusion,41, 182-194,2018,Elsevier

Carcillo, Fabrizio; Le Borgne, Yann-Aël; Caelen, Olivier; Bontempi, Gianluca. Streaming active learning strategies for real-life credit card fraud detection: assessment and visualization, International Journal of Data Science and Analytics, 5,4,285-300,2018,Springer International Publishing

Bertrand Lebichot, Yann-Aël Le Borgne, Liyun He, Frederic Oblé, Gianluca Bontempi Deep-Learning Domain Adaptation Techniques for Credit Cards Fraud Detection, INNSBDDL 2019: Recent Advances in Big Data and Deep Learning, pp 78-88, 2019

Fabrizio Carcillo, Yann-Aël Le Borgne, Olivier Caelen, Frederic Oblé, Gianluca Bontempi Combining Unsupervised and Supervised Learning in Credit Card Fraud Detection Information Sciences, 2019

# Approach

When implementing a fraud detection algorithm it's important to consider the following:
* what kind of output should the model give (fraud / no fraud classification, probability of fraud to be considered by a rule-based system)?
* how was the input data collected? Is it possible that the ground truth contains undetected frauds?
* what performance targets should the algorithm be optimized for (high precision, high recall, ...)?
* what is the consequence of not predicting a fraud / falsely predicting a fraud?

The dataset is very imbalanced, but a good choice of metric allows for meaningful comparison between models.
In a real application, it would be wise to identify a business related metric to optimize (expected profit of a decision).

The following steps are taken:
* explore the dataset: quantile comparisons and paired histograms suggest that some features are more relevant than others
* feature selection: we use a few techniques for choosing the candidate features
* learning curve analysis is used to estimate a reasonable split of the dataset
* logistic regression, random forests and gradient boosted trees are explored as viable algorithms

Next steps:
* hyperparameter tuning
* evaluation of a custom loss function (e.g. business-related)
* error analysis

# Data loading

```python
import pandas as pd
import numpy as np
import pylab as plt
import seaborn as sns

from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from sklearn_pandas import DataFrameMapper , CategoricalImputer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelBinarizer, LabelEncoder, RobustScaler
from sklearn.impute import SimpleImputer
import category_encoders as ce
from sklearn_pandas import DataFrameMapper , CategoricalImputer


import xgboost as xgb
from sklearn.model_selection import train_test_split


from sklearn.feature_selection import SelectKBest, SelectPercentile
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree  import DecisionTreeClassifier
from sklearn.linear_model  import RidgeClassifier

from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint as sp_randint , uniform
from scipy.stats import chi2_contingency

from sklearn.preprocessing import FunctionTransformer

from sklearn.model_selection import train_test_split, GridSearchCV,RandomizedSearchCV
from sklearn.model_selection import learning_curve, validation_curve, cross_val_score
from sklearn.model_selection import ShuffleSplit, StratifiedKFold

from sklearn.metrics import confusion_matrix, fbeta_score, make_scorer, average_precision_score, auc, \
    accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, classification_report, \
    brier_score_loss, roc_auc_score

from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer, MissingIndicator

import pickle

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import  DecisionTreeClassifier

import shap

```


```python
import df_helper as dfh
```


```python
import importlib
importlib.reload(dfh)
```




    <module 'df_helper' from 'c:\\DS\\CC_F\\df_helper.py'>




```python
# this database contains no null values
df= pd.read_csv('creditcard.csv',sep=',')
```

# Exploratory data analysis


```python
desc=df.describe().T
desc
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
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Time</th>
      <td>284807.0</td>
      <td>9.481386e+04</td>
      <td>47488.145955</td>
      <td>0.000000</td>
      <td>54201.500000</td>
      <td>84692.000000</td>
      <td>139320.500000</td>
      <td>172792.000000</td>
    </tr>
    <tr>
      <th>V1</th>
      <td>284807.0</td>
      <td>1.165980e-15</td>
      <td>1.958696</td>
      <td>-56.407510</td>
      <td>-0.920373</td>
      <td>0.018109</td>
      <td>1.315642</td>
      <td>2.454930</td>
    </tr>
    <tr>
      <th>V2</th>
      <td>284807.0</td>
      <td>3.416908e-16</td>
      <td>1.651309</td>
      <td>-72.715728</td>
      <td>-0.598550</td>
      <td>0.065486</td>
      <td>0.803724</td>
      <td>22.057729</td>
    </tr>
    <tr>
      <th>V3</th>
      <td>284807.0</td>
      <td>-1.373150e-15</td>
      <td>1.516255</td>
      <td>-48.325589</td>
      <td>-0.890365</td>
      <td>0.179846</td>
      <td>1.027196</td>
      <td>9.382558</td>
    </tr>
    <tr>
      <th>V4</th>
      <td>284807.0</td>
      <td>2.086869e-15</td>
      <td>1.415869</td>
      <td>-5.683171</td>
      <td>-0.848640</td>
      <td>-0.019847</td>
      <td>0.743341</td>
      <td>16.875344</td>
    </tr>
    <tr>
      <th>V5</th>
      <td>284807.0</td>
      <td>9.604066e-16</td>
      <td>1.380247</td>
      <td>-113.743307</td>
      <td>-0.691597</td>
      <td>-0.054336</td>
      <td>0.611926</td>
      <td>34.801666</td>
    </tr>
    <tr>
      <th>V6</th>
      <td>284807.0</td>
      <td>1.490107e-15</td>
      <td>1.332271</td>
      <td>-26.160506</td>
      <td>-0.768296</td>
      <td>-0.274187</td>
      <td>0.398565</td>
      <td>73.301626</td>
    </tr>
    <tr>
      <th>V7</th>
      <td>284807.0</td>
      <td>-5.556467e-16</td>
      <td>1.237094</td>
      <td>-43.557242</td>
      <td>-0.554076</td>
      <td>0.040103</td>
      <td>0.570436</td>
      <td>120.589494</td>
    </tr>
    <tr>
      <th>V8</th>
      <td>284807.0</td>
      <td>1.177556e-16</td>
      <td>1.194353</td>
      <td>-73.216718</td>
      <td>-0.208630</td>
      <td>0.022358</td>
      <td>0.327346</td>
      <td>20.007208</td>
    </tr>
    <tr>
      <th>V9</th>
      <td>284807.0</td>
      <td>-2.406455e-15</td>
      <td>1.098632</td>
      <td>-13.434066</td>
      <td>-0.643098</td>
      <td>-0.051429</td>
      <td>0.597139</td>
      <td>15.594995</td>
    </tr>
    <tr>
      <th>V10</th>
      <td>284807.0</td>
      <td>2.239751e-15</td>
      <td>1.088850</td>
      <td>-24.588262</td>
      <td>-0.535426</td>
      <td>-0.092917</td>
      <td>0.453923</td>
      <td>23.745136</td>
    </tr>
    <tr>
      <th>V11</th>
      <td>284807.0</td>
      <td>1.673327e-15</td>
      <td>1.020713</td>
      <td>-4.797473</td>
      <td>-0.762494</td>
      <td>-0.032757</td>
      <td>0.739593</td>
      <td>12.018913</td>
    </tr>
    <tr>
      <th>V12</th>
      <td>284807.0</td>
      <td>-1.254995e-15</td>
      <td>0.999201</td>
      <td>-18.683715</td>
      <td>-0.405571</td>
      <td>0.140033</td>
      <td>0.618238</td>
      <td>7.848392</td>
    </tr>
    <tr>
      <th>V13</th>
      <td>284807.0</td>
      <td>8.176030e-16</td>
      <td>0.995274</td>
      <td>-5.791881</td>
      <td>-0.648539</td>
      <td>-0.013568</td>
      <td>0.662505</td>
      <td>7.126883</td>
    </tr>
    <tr>
      <th>V14</th>
      <td>284807.0</td>
      <td>1.206296e-15</td>
      <td>0.958596</td>
      <td>-19.214325</td>
      <td>-0.425574</td>
      <td>0.050601</td>
      <td>0.493150</td>
      <td>10.526766</td>
    </tr>
    <tr>
      <th>V15</th>
      <td>284807.0</td>
      <td>4.913003e-15</td>
      <td>0.915316</td>
      <td>-4.498945</td>
      <td>-0.582884</td>
      <td>0.048072</td>
      <td>0.648821</td>
      <td>8.877742</td>
    </tr>
    <tr>
      <th>V16</th>
      <td>284807.0</td>
      <td>1.437666e-15</td>
      <td>0.876253</td>
      <td>-14.129855</td>
      <td>-0.468037</td>
      <td>0.066413</td>
      <td>0.523296</td>
      <td>17.315112</td>
    </tr>
    <tr>
      <th>V17</th>
      <td>284807.0</td>
      <td>-3.800113e-16</td>
      <td>0.849337</td>
      <td>-25.162799</td>
      <td>-0.483748</td>
      <td>-0.065676</td>
      <td>0.399675</td>
      <td>9.253526</td>
    </tr>
    <tr>
      <th>V18</th>
      <td>284807.0</td>
      <td>9.572133e-16</td>
      <td>0.838176</td>
      <td>-9.498746</td>
      <td>-0.498850</td>
      <td>-0.003636</td>
      <td>0.500807</td>
      <td>5.041069</td>
    </tr>
    <tr>
      <th>V19</th>
      <td>284807.0</td>
      <td>1.039817e-15</td>
      <td>0.814041</td>
      <td>-7.213527</td>
      <td>-0.456299</td>
      <td>0.003735</td>
      <td>0.458949</td>
      <td>5.591971</td>
    </tr>
    <tr>
      <th>V20</th>
      <td>284807.0</td>
      <td>6.406703e-16</td>
      <td>0.770925</td>
      <td>-54.497720</td>
      <td>-0.211721</td>
      <td>-0.062481</td>
      <td>0.133041</td>
      <td>39.420904</td>
    </tr>
    <tr>
      <th>V21</th>
      <td>284807.0</td>
      <td>1.656562e-16</td>
      <td>0.734524</td>
      <td>-34.830382</td>
      <td>-0.228395</td>
      <td>-0.029450</td>
      <td>0.186377</td>
      <td>27.202839</td>
    </tr>
    <tr>
      <th>V22</th>
      <td>284807.0</td>
      <td>-3.444850e-16</td>
      <td>0.725702</td>
      <td>-10.933144</td>
      <td>-0.542350</td>
      <td>0.006782</td>
      <td>0.528554</td>
      <td>10.503090</td>
    </tr>
    <tr>
      <th>V23</th>
      <td>284807.0</td>
      <td>2.578648e-16</td>
      <td>0.624460</td>
      <td>-44.807735</td>
      <td>-0.161846</td>
      <td>-0.011193</td>
      <td>0.147642</td>
      <td>22.528412</td>
    </tr>
    <tr>
      <th>V24</th>
      <td>284807.0</td>
      <td>4.471968e-15</td>
      <td>0.605647</td>
      <td>-2.836627</td>
      <td>-0.354586</td>
      <td>0.040976</td>
      <td>0.439527</td>
      <td>4.584549</td>
    </tr>
    <tr>
      <th>V25</th>
      <td>284807.0</td>
      <td>5.340915e-16</td>
      <td>0.521278</td>
      <td>-10.295397</td>
      <td>-0.317145</td>
      <td>0.016594</td>
      <td>0.350716</td>
      <td>7.519589</td>
    </tr>
    <tr>
      <th>V26</th>
      <td>284807.0</td>
      <td>1.687098e-15</td>
      <td>0.482227</td>
      <td>-2.604551</td>
      <td>-0.326984</td>
      <td>-0.052139</td>
      <td>0.240952</td>
      <td>3.517346</td>
    </tr>
    <tr>
      <th>V27</th>
      <td>284807.0</td>
      <td>-3.666453e-16</td>
      <td>0.403632</td>
      <td>-22.565679</td>
      <td>-0.070840</td>
      <td>0.001342</td>
      <td>0.091045</td>
      <td>31.612198</td>
    </tr>
    <tr>
      <th>V28</th>
      <td>284807.0</td>
      <td>-1.220404e-16</td>
      <td>0.330083</td>
      <td>-15.430084</td>
      <td>-0.052960</td>
      <td>0.011244</td>
      <td>0.078280</td>
      <td>33.847808</td>
    </tr>
    <tr>
      <th>Amount</th>
      <td>284807.0</td>
      <td>8.834962e+01</td>
      <td>250.120109</td>
      <td>0.000000</td>
      <td>5.600000</td>
      <td>22.000000</td>
      <td>77.165000</td>
      <td>25691.160000</td>
    </tr>
    <tr>
      <th>Class</th>
      <td>284807.0</td>
      <td>1.727486e-03</td>
      <td>0.041527</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



We notice a number of outliers in the numerical variables. This may be a challenge for some algorithms.




```python
df.Class.value_counts()
```




    0    284315
    1       492
    Name: Class, dtype: int64



Given that the target class is binary the model will be performing binary prediction, also known as *binary classification*.
As expected the database is strongly imbalanced. 

## Class Imbalance

There are several ways to deal with class imbalance. In this case we will **NOT**:
* undersample the majority class
* oversample with repetition the minority class
* try to generate synthetic data (e.g. SMOTE technique)
* recast the problem as anomaly detection

We will focus particularly on algorithm that can natively handle imbalance (e.g. **tree**-based algorithms).

Furthermore we will **not** use accuracy as performance metric but the **area under the precision-recall curve, AUC, and Expected Cost**.


## PCA-Reduced Numerical Features (V1-V28) 

### Outliers


```python
cols=['V{}'.format(i) for i in range(1,29)]

plt.figure(figsize=(15,4))
plt.plot(desc.loc[cols]['min'],'ko-',label='min')
plt.plot(desc.loc[cols]['mean'],'ro-',label='mean')
plt.plot(desc.loc[cols]['max'],'ko-',label='max')

plt.plot(desc.loc[cols]['25%'],'gd--',label='25%',alpha=0.5)
plt.plot(desc.loc[cols]['75%'],'bd--',label='75%',alpha=0.5)

plt.legend()
plt.title('there are significant outliers but most datapoints are very close to the mean');
```


{% raw %}![alt](/assets/CC_fraud/output_12_0.png){% endraw %}


We notice that the outlier value are much larger than the 25-75% quantile range


```python
desc0=df[df.Class==0].describe().T
desc1=df[df.Class==1].describe().T


plt.figure(figsize=(15,10))
plt.subplot(211)
plt.plot(desc0.loc[cols]['min'],'ko-',label='min')
plt.plot(desc0.loc[cols]['mean'],'ro-',label='mean')
plt.plot(desc0.loc[cols]['max'],'ko-',label='max')

plt.plot(desc0.loc[cols]['25%'],'gd--',label='25%',alpha=0.5)
plt.plot(desc0.loc[cols]['75%'],'bd--',label='75%',alpha=0.5)

plt.legend()
plt.title('Non Frauds');


# plt.figure(figsize=(15,4))
plt.subplot(212)
plt.plot(desc1.loc[cols]['min'],'ko-',label='min')
plt.plot(desc1.loc[cols]['mean'],'ro-',label='mean')
plt.plot(desc1.loc[cols]['max'],'ko-',label='max')

plt.plot(desc1.loc[cols]['25%'],'gd--',label='25%',alpha=0.5)
plt.plot(desc1.loc[cols]['75%'],'bd--',label='75%',alpha=0.5)

plt.legend()
plt.title('Frauds');
```


{% raw %}![alt](/assets/CC_fraud/output_14_0.png){% endraw %}


Interestingly class 1 (Fraud) does not exhibit such strong outliers.


```python
plt.figure(figsize=(15,4))
plt.fill_between(cols, desc0.loc[cols]['25%'],desc0.loc[cols]['75%'], alpha=0.1,color="g", label='Non Frauds')

plt.fill_between(cols, desc1.loc[cols]['25%'],desc1.loc[cols]['75%'], alpha=0.1,color="r", label='Frauds')
plt.title('25-75 Quantile')
plt.legend();
```


{% raw %}![alt](/assets/CC_fraud/output_16_0.png){% endraw %}


The 25-75 quantile of variables V1-18 seems quite informative with the exceptions of V5, V6, V8, V13, V15.

### Time


```python
# we can bin the time by hour of the two days included in our dataset
g = sns.FacetGrid(df, hue="Class",height=3, aspect=1.8 )
g.map(sns.distplot, "Time", hist=True, kde=False,norm_hist=True, bins=list(range(0,172800,3600)) )
g.add_legend();
plt.title('Frauds have a different time distribution');
```


{% raw %}![alt](/assets/CC_fraud/output_19_0.png){% endraw %}



```python
# and then compare the same hour across the two days contained in the dataset
df['hour_c']=pd.cut(df.Time, 48,labels=list(range(0,48))).astype(int)
df['hour']=df['hour_c']%24
df['day']=df['hour_c']//24
```


```python
g = sns.FacetGrid(df, hue="Class",height=3, aspect=1.8 )
g.map(sns.distplot, "hour", hist=True, kde=False,norm_hist=True, bins=list(range(0,24)) )
g.add_legend();
plt.title('Frauds have a different hour distribution');
```


{% raw %}![alt](/assets/CC_fraud/output_21_0.png){% endraw %}



```python
g = sns.FacetGrid(df, col='day',hue="Class",height=3, aspect=1.8 )
g.map(sns.distplot, "hour", hist=True, kde=False,norm_hist=True, bins=list(range(0,24)) )
g.add_legend();
plt.suptitle('Fraud trends are not too dissimilar over the two days');
```


{% raw %}![alt](/assets/CC_fraud/output_22_0.png){% endraw %}


We lack sufficient evidence to claim the the hourly behavior across the two days is different.

### Amount


```python
sns.distplot(df.Amount);
```


{% raw %}![alt](/assets/CC_fraud/output_25_0.png){% endraw %}


We need to clip the Y-Axis otherwise the large outliers of class 0 make the following figures difficult to read.


```python
g = sns.FacetGrid(df, hue="Class",height=3, aspect=1.8 )
g.map(sns.distplot, "Amount", hist=True, kde=False,norm_hist=True)# , bins=list(range(0,175000,10000)) )
g.add_legend();
plt.title('Frauds have a different distribution in terms of amount');
plt.axis([0,2000,0,0.005]);
```


{% raw %}![alt](/assets/CC_fraud/output_27_0.png){% endraw %}


Most frauds seem to occur for small amounts. 


```python
sns.boxplot(data=df,y='Amount',x='Class');
# plt.axis([-1,2,-10,1000]);
plt.yscale('Log')
```


{% raw %}![alt](/assets/CC_fraud/output_29_0.png){% endraw %}


## Correlation Analysis

The 25-75Q range is quite different up to feature V18. 


```python
fig=plt.figure(figsize=(12,12))
ax=plt.gca()
corr=df.corr()
sns.heatmap(corr,ax=ax, 
        xticklabels=corr.columns,
        yticklabels=corr.columns, cmap="gnuplot2");
```


{% raw %}![alt](/assets/CC_fraud/output_32_0.png){% endraw %}


There is no correlation between the features obtained by PCA, as expected.


```python
fig=plt.figure(figsize=(12,12))
ax=plt.gca()
corr=df[cols].corr()
sns.heatmap(corr,ax=ax, 
        xticklabels=corr.columns,
        yticklabels=corr.columns, cmap="gnuplot2",vmin=-0.02,vmax=0.02);
```


{% raw %}![alt](/assets/CC_fraud/output_34_0.png){% endraw %}


## Paired Histograms


```python
plt.figure(figsize=(18,24))
c=1
cols=['V{}'.format(i) for i in range(1,29)]
for col in cols:
    ax=plt.subplot(7,4,c)
    dfh.compare_histograms(df,col,'Class',density=True,values=[0,1],quantiles=[0.25,0.75]);
    plt.xlabel('')
    ax.text(.5,.9,col,
        horizontalalignment='center',
        transform=ax.transAxes)

    c+=1
```


{% raw %}![alt](/assets/CC_fraud/output_36_0.png){% endraw %}


While some features show remarkable difference in the distributions, others are very similar (i.e. V13, V15).

# Feature Selection

The EDA suggests that some features may not be important. Even though many algorithms support regularization techniques discarding uninformative features may improve the prediction.


## Select K-best technique


```python
# selection using the ANOVA F-value between label/feature for classification tasks, considering the features are numerical and can mostly be positive and negative.

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif 
cols=['V{}'.format(i) for i in range(1,29)]
cols+=['Time','Amount']

sel_AF= SelectKBest(f_classif, k='all').fit(df[cols ],df['Class'])

```


```python
ncols=np.array(cols )
sel_kb=list(ncols[sel_AF.get_support()])

scores_ANOVAf = -np.log10(sel_AF.pvalues_+1e-50)
scores_ANOVAf /= scores_ANOVAf.max()

plt.figure(figsize=(15,4))
plt.bar(ncols, scores_ANOVAf, width=.2, label=r'Univariate score ($-Log(p_{value})$)')
plt.legend();
```


{% raw %}![alt](/assets/CC_fraud/output_40_0.png){% endraw %}


As expected V13 and V15 have low predictivity.

## Recursive feature elimination with cross-validation

Recursive feature selections allows to determinte the optimal amount of features.



```python
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.tree import DecisionTreeClassifier
    
rs=RobustScaler(quantile_range=(25.0, 75.0))
X=rs.fit_transform(df[cols])
y=df['Class']

# cl = LogisticRegression(class_weight='balanced',max_iter=1000,solver='newton-cg') # very slow, difficult convergence
cl = DecisionTreeClassifier(class_weight='balanced')

rfecv = RFECV(estimator=cl, step=2, cv=StratifiedKFold(2),
              scoring='accuracy',n_jobs=-1)
rfecv.fit(X,y)

print("Optimal number of features : %d" % rfecv.n_features_)

# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()
```

    Optimal number of features : 6
    


{% raw %}![alt](/assets/CC_fraud/output_43_1.png){% endraw %}



```python
sel_RFE=list(ncols[rfecv.get_support()])
```


```python
plt.figure(figsize=(15,4))

plt.bar(ncols,rfecv.ranking_ , width=.2, label='ranking (lowest is best)')
plt.legend();
```


{% raw %}![alt](/assets/CC_fraud/output_45_0.png){% endraw %}



```python
plt.figure(figsize=(15,4))
scores_RFE=1./rfecv.ranking_ 
scores_RFE=scores_RFE/scores_RFE.max()
plt.bar(ncols,scores_RFE , width=.2, align='edge', label='1/ranking (RFE)')
plt.bar(ncols, scores_ANOVAf, width=-.2,  align='edge',label=r'Univariate score ($-Log(p_{value})$)')
plt.legend();
```


{% raw %}![alt](/assets/CC_fraud/output_46_0.png){% endraw %}


Due to time constraint considerations we are forced to use a fast but simple method for RFECV.



# Modeling

## Learning curve analysis


```python
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression

rs=RobustScaler(quantile_range=(25.0, 75.0))
X=rs.fit_transform(df[cols])
y=df['Class']

estimator= LogisticRegression(C=0.1,penalty='l2',class_weight='balanced',solver='lbfgs', verbose=0,max_iter=100)
dfh.plot_learning_curve(estimator, 'LR', X, y, ylim=None, cv=5,
                        n_jobs=4, train_sizes=np.linspace(.3, 1.0, 10), scoring='roc_auc')
```




    <module 'matplotlib.pyplot' from 'c:\\anaconda3\\envs\\ds\\lib\\site-packages\\matplotlib\\pyplot.py'>




{% raw %}![alt](/assets/CC_fraud/output_49_1.png){% endraw %}


The size of the dataset is sufficient to achieve a good prediction (metrics is ROC-AUC).

Using the learning curve we use 70% of the data for the training set and split the remainder 60-40 between test and validation set.



```python
train_size=0.7
test_size=0.6
y=df['Class']
cols=list(set(df.columns)-set(['Class']))
X_train1,X_r,y_train,y_r= train_test_split(df[cols] , y, train_size=train_size , )
X_test1,X_val1,y_test,y_val= train_test_split(X_r, y_r, train_size=test_size,    )

print('train: ', len(X_train1))
print('test: ', len(X_test1))
print('val: ', len(X_val1))

```

    train:  199364
    test:  51265
    val:  34178
    

Given the abundance of outliers we use a RobustScaler in the 25-75 quantile range.


```python
# Robust scaling

rs=RobustScaler(quantile_range=(25.0, 75.0))
X_train=rs.fit_transform(X_train1)
X_test=rs.transform(X_test1)
X_val=rs.transform(X_val1)

X_train=pd.DataFrame(X_train,columns=X_train1.columns)
X_test=pd.DataFrame(X_test,columns=X_test1.columns)
X_val=pd.DataFrame(X_val,columns=X_test1.columns)

```

## Baseline

### LogisticRegression

LogisticRegression is a fast, decent performance method that can be a very valuable baseline.


```python
y_train.value_counts()
```




    0    199018
    1       346
    Name: Class, dtype: int64




```python
imbalance_correction=y_train.value_counts()[0]/y_train.value_counts()[1]
imbalance_correction
```




    575.1965317919075




```python
lr_cl = LogisticRegression(C=0.1,penalty='l2',solver='lbfgs',class_weight='balanced',verbose=0,max_iter=1000)
lr_cl.fit(X_train,y_train)

# Predict the labels of the test set: preds
y_pred_lr = lr_cl.predict(X_test)
y_score_lr = lr_cl.predict_proba(X_test)[:,1]

rf=dfh.add_metrics(y_test,y_score_lr,y_pred_lr,'LR L2 bal')
rf.iloc[:,0:8]
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
      <th>model</th>
      <th>tn</th>
      <th>fp</th>
      <th>fn</th>
      <th>tp</th>
      <th>P</th>
      <th>R</th>
      <th>AP</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>LR L2 bal</td>
      <td>49818</td>
      <td>1367</td>
      <td>7</td>
      <td>73</td>
      <td>0.050694</td>
      <td>0.9125</td>
      <td>0.657817</td>
    </tr>
  </tbody>
</table>
</div>




```python
# we repeat the fitting without automatic correction of imbalance

lr_cl = LogisticRegression(C=0.1,penalty='l2',solver='lbfgs',verbose=0,max_iter=1000)
lr_cl.fit(X_train,y_train)

# Predict the labels of the test set: preds
y_pred_lr = lr_cl.predict(X_test)
y_score_lr = lr_cl.predict_proba(X_test)[:,1]

rf=dfh.add_metrics(y_test,y_score_lr,y_pred_lr,'LR L2 unbal',rf)
rf.iloc[:,0:8]
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
      <th>model</th>
      <th>tn</th>
      <th>fp</th>
      <th>fn</th>
      <th>tp</th>
      <th>P</th>
      <th>R</th>
      <th>AP</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>LR L2 bal</td>
      <td>49818</td>
      <td>1367</td>
      <td>7</td>
      <td>73</td>
      <td>0.050694</td>
      <td>0.9125</td>
      <td>0.657817</td>
    </tr>
    <tr>
      <th>1</th>
      <td>LR L2 unbal</td>
      <td>51174</td>
      <td>11</td>
      <td>30</td>
      <td>50</td>
      <td>0.819672</td>
      <td>0.6250</td>
      <td>0.729219</td>
    </tr>
  </tbody>
</table>
</div>



It could be that the imbalance correction was eccessive, pushing toward a very high recall but low precision.
We can manually adjust the correction.



```python
# we repeat the fitting without automatic correction of imbalance
correction_factor=20

lr_cl = LogisticRegression(C=0.1,penalty='l2',solver='lbfgs',verbose=0,max_iter=1000, class_weight={0:1,1:imbalance_correction/correction_factor})
lr_cl.fit(X_train,y_train)

y_pred_lr = lr_cl.predict(X_test)
y_score_lr = lr_cl.predict_proba(X_test)[:,1]

rf=dfh.add_metrics(y_test,y_score_lr,y_pred_lr,'LR L2 bal/%d' % correction_factor,rf)
rf.iloc[:,0:8]
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
      <th>model</th>
      <th>tn</th>
      <th>fp</th>
      <th>fn</th>
      <th>tp</th>
      <th>P</th>
      <th>R</th>
      <th>AP</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>LR L2 bal</td>
      <td>49818</td>
      <td>1367</td>
      <td>7</td>
      <td>73</td>
      <td>0.050694</td>
      <td>0.9125</td>
      <td>0.657817</td>
    </tr>
    <tr>
      <th>1</th>
      <td>LR L2 unbal</td>
      <td>51174</td>
      <td>11</td>
      <td>30</td>
      <td>50</td>
      <td>0.819672</td>
      <td>0.6250</td>
      <td>0.729219</td>
    </tr>
    <tr>
      <th>2</th>
      <td>LR L2 bal/20</td>
      <td>51137</td>
      <td>48</td>
      <td>15</td>
      <td>65</td>
      <td>0.575221</td>
      <td>0.8125</td>
      <td>0.720516</td>
    </tr>
  </tbody>
</table>
</div>



Adjusting the class weight is similar to tuning the classifier threshold to tradeoff true and false positives.


```python
lr_par=pd.DataFrame(lr_cl.coef_/np.max(lr_cl.coef_),columns=X_train.columns)
lr_par=lr_par[(lr_par>0)].dropna(axis=1).T

lr_par.columns=['LR L2']
lr_par.T
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
      <th>V5</th>
      <th>Amount</th>
      <th>V21</th>
      <th>V4</th>
      <th>V24</th>
      <th>V2</th>
      <th>V22</th>
      <th>V25</th>
      <th>V1</th>
      <th>hour</th>
      <th>V28</th>
      <th>V19</th>
      <th>V11</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>LR L2</th>
      <td>0.304461</td>
      <td>0.117131</td>
      <td>0.100901</td>
      <td>1.0</td>
      <td>0.080116</td>
      <td>0.135972</td>
      <td>0.691141</td>
      <td>0.003287</td>
      <td>0.335956</td>
      <td>0.629979</td>
      <td>0.008809</td>
      <td>0.039898</td>
      <td>0.075912</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.figure(figsize=(15,4))
plt.bar(lr_par.index,lr_par['LR L2'],label='LR L2')

plt.legend();
```


{% raw %}![alt](/assets/CC_fraud/output_63_0.png){% endraw %}


We notice that even with L2 regularization there was a strong feature selection.
We compare the result with L1 regularization


```python
lr_cl = LogisticRegression(C=0.1,penalty='l1',class_weight='balanced',solver='liblinear',verbose=0,max_iter=1000)

lr_cl.fit(X_train,y_train)

# Predict the labels of the test set: preds
y_pred = lr_cl.predict(X_test)
y_score = lr_cl.predict_proba(X_test)[:,1]

rf=dfh.add_metrics(y_test,y_score,y_pred,'LR L1',rf)
rf.iloc[:,0:8]
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
      <th>model</th>
      <th>tn</th>
      <th>fp</th>
      <th>fn</th>
      <th>tp</th>
      <th>P</th>
      <th>R</th>
      <th>AP</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>LR L2 bal</td>
      <td>49818</td>
      <td>1367</td>
      <td>7</td>
      <td>73</td>
      <td>0.050694</td>
      <td>0.9125</td>
      <td>0.657817</td>
    </tr>
    <tr>
      <th>1</th>
      <td>LR L2 unbal</td>
      <td>51174</td>
      <td>11</td>
      <td>30</td>
      <td>50</td>
      <td>0.819672</td>
      <td>0.6250</td>
      <td>0.729219</td>
    </tr>
    <tr>
      <th>2</th>
      <td>LR L2 bal/20</td>
      <td>51137</td>
      <td>48</td>
      <td>15</td>
      <td>65</td>
      <td>0.575221</td>
      <td>0.8125</td>
      <td>0.720516</td>
    </tr>
    <tr>
      <th>3</th>
      <td>LR L1</td>
      <td>49819</td>
      <td>1366</td>
      <td>7</td>
      <td>73</td>
      <td>0.050730</td>
      <td>0.9125</td>
      <td>0.663531</td>
    </tr>
  </tbody>
</table>
</div>




```python
lr_par2=pd.DataFrame(lr_cl.coef_/np.max(lr_cl.coef_),columns=X_train.columns)
lr_par2=lr_par2[(lr_par2>0)].dropna(axis=1).T

lr_par2.columns=['LR L1']
lr_par2.T
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
      <th>V3</th>
      <th>V5</th>
      <th>Amount</th>
      <th>V4</th>
      <th>V24</th>
      <th>V2</th>
      <th>V22</th>
      <th>V23</th>
      <th>V1</th>
      <th>hour</th>
      <th>V28</th>
      <th>V27</th>
      <th>V19</th>
      <th>V11</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>LR L1</th>
      <td>0.11628</td>
      <td>0.725048</td>
      <td>0.410215</td>
      <td>0.760777</td>
      <td>0.061369</td>
      <td>0.568629</td>
      <td>0.642305</td>
      <td>0.07543</td>
      <td>1.0</td>
      <td>0.428603</td>
      <td>0.082954</td>
      <td>0.032554</td>
      <td>0.279705</td>
      <td>0.406197</td>
    </tr>
  </tbody>
</table>
</div>




```python
cmp_par=lr_par.join(lr_par2).fillna(0)
cmp_par.plot.bar();
```


{% raw %}![alt](/assets/CC_fraud/output_67_0.png){% endraw %}


As expected the L1 model keeps less features.


```python
print("accuracy_score: ",accuracy_score(y_test,y_pred))
print("balanced_accuracy_score: ",balanced_accuracy_score(y_test,y_pred))
print("ROC AUC: ",roc_auc_score(y_test,y_score))
print()
print(classification_report(y_test,y_pred))
```

    accuracy_score:  0.9732175948502877
    balanced_accuracy_score:  0.9429062469473479
    ROC AUC:  0.9798292956920973
    
                  precision    recall  f1-score   support
    
               0       1.00      0.97      0.99     51185
               1       0.05      0.91      0.10        80
    
        accuracy                           0.97     51265
       macro avg       0.53      0.94      0.54     51265
    weighted avg       1.00      0.97      0.99     51265
    
    


```python
dfh.plot_pr_mat(rf)
plt.legend(loc=3);
```


{% raw %}![alt](/assets/CC_fraud/output_70_0.png){% endraw %}


The area under the precision-recall curve is a good way to compare the models.


```python
dfh.plot_roc_mat(rf)
```


{% raw %}![alt](/assets/CC_fraud/output_72_0.png){% endraw %}






## Gradient Boosted Trees
While the performance of Logistic Regression is reasonably acceptable (we can detect 95% of frauds with a false positive rate of 10%), we have a look at more sophisticated algorithms, such as XGBoost.  


```python
vc=y_train.value_counts().values
weight=vc[0]/vc[1]
print(weight)

# in fact the performance is better without class weights
xg_cl  = xgb.XGBClassifier(objective='binary:logistic', n_estimators=10, seed=123, scale_pos_weight=1)

# Fit the classifier to the training set
xg_cl.fit(X_train,y_train)

# Predict the labels of the test set: preds
y_pred_xg = xg_cl.predict(X_test)
y_score_xg = xg_cl.predict_proba(X_test)[:,1]

rf=dfh.add_metrics(y_test,y_score_xg,y_pred_xg,'xgb',rf)
rf.iloc[:,0:8]
```

    575.1965317919075
    




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
      <th>model</th>
      <th>tn</th>
      <th>fp</th>
      <th>fn</th>
      <th>tp</th>
      <th>P</th>
      <th>R</th>
      <th>AP</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>LR L2 bal</td>
      <td>49818</td>
      <td>1367</td>
      <td>7</td>
      <td>73</td>
      <td>0.050694</td>
      <td>0.9125</td>
      <td>0.657817</td>
    </tr>
    <tr>
      <th>1</th>
      <td>LR L2 unbal</td>
      <td>51174</td>
      <td>11</td>
      <td>30</td>
      <td>50</td>
      <td>0.819672</td>
      <td>0.6250</td>
      <td>0.729219</td>
    </tr>
    <tr>
      <th>2</th>
      <td>LR L2 bal/20</td>
      <td>51137</td>
      <td>48</td>
      <td>15</td>
      <td>65</td>
      <td>0.575221</td>
      <td>0.8125</td>
      <td>0.720516</td>
    </tr>
    <tr>
      <th>3</th>
      <td>LR L1</td>
      <td>49819</td>
      <td>1366</td>
      <td>7</td>
      <td>73</td>
      <td>0.050730</td>
      <td>0.9125</td>
      <td>0.663531</td>
    </tr>
    <tr>
      <th>4</th>
      <td>xgb</td>
      <td>51181</td>
      <td>4</td>
      <td>18</td>
      <td>62</td>
      <td>0.939394</td>
      <td>0.7750</td>
      <td>0.819451</td>
    </tr>
  </tbody>
</table>
</div>




```python
xg_cl  = RandomForestClassifier(n_estimators=20, class_weight="balanced")

# Fit the classifier to the training set
xg_cl.fit(X_train,y_train)

# Predict the labels of the test set: preds
y_pred_rf = xg_cl.predict(X_test)
y_score_rf = xg_cl.predict_proba(X_test)[:,1]

rf=dfh.add_metrics(y_test,y_score_rf,y_pred_rf,'RF',rf)
rf.iloc[:,0:8]
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
      <th>model</th>
      <th>tn</th>
      <th>fp</th>
      <th>fn</th>
      <th>tp</th>
      <th>P</th>
      <th>R</th>
      <th>AP</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>LR L2 bal</td>
      <td>49818</td>
      <td>1367</td>
      <td>7</td>
      <td>73</td>
      <td>0.050694</td>
      <td>0.9125</td>
      <td>0.657817</td>
    </tr>
    <tr>
      <th>1</th>
      <td>LR L2 unbal</td>
      <td>51174</td>
      <td>11</td>
      <td>30</td>
      <td>50</td>
      <td>0.819672</td>
      <td>0.6250</td>
      <td>0.729219</td>
    </tr>
    <tr>
      <th>2</th>
      <td>LR L2 bal/20</td>
      <td>51137</td>
      <td>48</td>
      <td>15</td>
      <td>65</td>
      <td>0.575221</td>
      <td>0.8125</td>
      <td>0.720516</td>
    </tr>
    <tr>
      <th>3</th>
      <td>LR L1</td>
      <td>49819</td>
      <td>1366</td>
      <td>7</td>
      <td>73</td>
      <td>0.050730</td>
      <td>0.9125</td>
      <td>0.663531</td>
    </tr>
    <tr>
      <th>4</th>
      <td>xgb</td>
      <td>51181</td>
      <td>4</td>
      <td>18</td>
      <td>62</td>
      <td>0.939394</td>
      <td>0.7750</td>
      <td>0.819451</td>
    </tr>
    <tr>
      <th>5</th>
      <td>RF</td>
      <td>51184</td>
      <td>1</td>
      <td>19</td>
      <td>61</td>
      <td>0.983871</td>
      <td>0.7625</td>
      <td>0.825220</td>
    </tr>
  </tbody>
</table>
</div>




```python
dfh.plot_pr_mat(rf)
```


{% raw %}![alt](/assets/CC_fraud/output_84_1.png){% endraw %}


XGBoost and RandomForest result in a significantly better performance than Logistic Regression.

# Next steps

Next steps that could be performed would be:
* hyperparameter tuning
* evaluation of a custom loss function (e.g. business-related)
* error analysis
