---
title: "CC 3: Fairness"
permalink: /credit_card/cc_fairness/
excerpt: "Credit Cards: Fairness"
last_modified_at: 2020-08-11T15:53:52-04:00
toc: true
toc_label: "Page Index"
toc_sticky: true
author_profile: false
classes: post
tags:
  - Classification
  - Fairness
categories:
  - Credit Card
---

# Fairness analysis


Credit card application is a typical case for the assessment of fairness in the model, e.g. if the genders are treated in the same way.
Addressing fairness issues in this database is probably going to be problematic because of the small database size.



```python
# stratify=True: Singleton array array(True) cannot be considered a valid collection.
X_train,X_test,X_val, y_train,y_test,y_val=apply_mapper(cf, mapper, f_num, f_num_log, f_bin, f_cat, train_size=400,test_size=120, random_state=42, stratify=None)


```


```python
def assign_set(cf,X_train,X_test,X_val):
    # see which set each sample is assigned to
    
    ef=cf.copy()
    ef['set']=''
    
    ef.loc[X_train.index.values,'set']='train'
    ef.loc[(X_test.index),'set']='test'
    ef.loc[(X_val.index),'set']='val'
    
    return ef

ef=assign_set(cf,X_train,X_test,X_val)
ef['IncomeLog']=np.log10(ef['Income']+1)
```


```python
def plot_balance(ef,var):
    "plots the unbalance of feature /label var across sets"
    
    R=pd.DataFrame( ef.groupby(by=['set','Gender'])[var].value_counts(normalize=True)[:,:,True] )
    R=pd.DataFrame(R.to_records())
    sns.barplot(x='Gender',y=var, hue='set' ,data=R)
    plt.legend(loc=3)
    #     plt.ylim(0.0,0.55);
    return
```


```python
var='ApprovalStatus'
plot_balance(ef,var)
```

{% raw %}![alt](/assets/UCI_Credit_Cards_fairness/output_65_0.png){% endraw %}
{% raw %}![alt](/assets/UCI_Credit_Cards_fairness/output_13_0.png){% endraw %}


The plot shows that the splits we generated are not perfectly balanced in terms of apporvals by gender.
Please note that the gender variable is assigned True/False and it is impossible to identify the gender it refers to.
Overrepresentation of one class may lead to unbalanced outcomes.

## Visualization of dataset gender imbalance 

Therefore we check if the features show a balanced split for the genders.
Ratios are used to compare the binary features.


```python
var='PriorDefault'
plot_balance(ef,var)
```


{% raw %}![alt](/assets/UCI_Credit_Cards_fairness/output_15_0.png){% endraw %}



```python
var='Employed'
plot_balance(ef,var)
```


{% raw %}![alt](/assets/UCI_Credit_Cards_fairness/output_16_0.png){% endraw %}


Histograms are used to compare numerical features.


```python
var='YearsEmployed'
g = sns.FacetGrid(ef,hue="set",height=5) #col='set', 
# g = (g.map(sns.distplot, var, hist=False, rug=True))
g = (g.map(sns.distplot, var, hist=True, kde=False,hist_kws={'histtype':'step','lw':3,'density':True}))
plt.legend(loc=1);

```


{% raw %}![alt](/assets/UCI_Credit_Cards_fairness/output_18_0.png){% endraw %}



```python
var='YearsEmployed'
g = sns.FacetGrid(ef,hue="set",height=4, col='set',col_order=['train','test','val'])
g = (g.map(sns.distplot, var, hist=True, rug=True))
```


{% raw %}![alt](/assets/UCI_Credit_Cards_fairness/output_19_0.png){% endraw %}



```python
var='IncomeLog'
g = sns.FacetGrid(ef,hue="set",height=4, col='set',col_order=['train','test','val'])
g = (g.map(sns.distplot, var, hist=True, rug=True))
```


{% raw %}![alt](/assets/UCI_Credit_Cards_fairness/output_20_0.png){% endraw %}



```python
var='IncomeLog'
g = sns.FacetGrid(ef,hue="set",height=5) #col='set', 
g = (g.map(sns.distplot, var, hist=True, kde=False,hist_kws={'histtype':'step','lw':3,'density':True}))
plt.legend(loc=1);
```


{% raw %}![alt](/assets/UCI_Credit_Cards_fairness/output_21_0.png){% endraw %}



```python
var='Age'
g = sns.FacetGrid(ef,hue="set",height=5) #col='set', 
g = (g.map(sns.distplot, var, hist=True, kde=False,hist_kws={'histtype':'step','lw':3,'density':True}))
plt.legend(loc=1);
```


{% raw %}![alt](/assets/UCI_Credit_Cards_fairness/output_22_0.png){% endraw %}


As remarked in section 2, it's not possible obtain a well stratified dataset split.
We expect therefore issues in terms of gender fairness.




## Fairness of classifier

Fairness analysis can be conducted by fitting a classifier, then evaluating the scores for each individual gender class.


```python
xg_cl = xgb.XGBClassifier(objective='binary:logistic', n_estimators=10, seed=123)

# Fit the classifier to the training set
xg_cl.fit(X_train,y_train)

# Predict the labels of the test set: preds
y_pred = xg_cl.predict(X_test)
y_score = xg_cl.predict_proba(X_test)[:,1]

rf=dfh.add_metrics(y_test,y_score,y_pred,'xgb all')
```


```python
for g in [0,1]:

    xt=X_test[(X_test.Gender==g)]
    yt=y_test[(X_test.Gender==g)]

    y_pred_g = xg_cl.predict(xt)
    y_score_g = xg_cl.predict_proba(xt)[:,1]

    rf=dfh.add_metrics(yt,y_score_g,y_pred_g,'g={}'.format(g),rf )
```


```python
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
      <td>xgb all</td>
      <td>52</td>
      <td>12</td>
      <td>7</td>
      <td>49</td>
      <td>0.80</td>
      <td>0.88</td>
      <td>0.91</td>
    </tr>
    <tr>
      <th>1</th>
      <td>g=0</td>
      <td>38</td>
      <td>9</td>
      <td>6</td>
      <td>31</td>
      <td>0.78</td>
      <td>0.84</td>
      <td>0.89</td>
    </tr>
    <tr>
      <th>2</th>
      <td>g=1</td>
      <td>14</td>
      <td>3</td>
      <td>1</td>
      <td>18</td>
      <td>0.86</td>
      <td>0.95</td>
      <td>0.95</td>
    </tr>
  </tbody>
</table>
</div>


### Confusion matrix

The model has lower precision and recall for gender 0 than for gender 1.
More than 12% of applicants in gender 0 are incorrectly denied a credit card, versus 5% of gender 1.
Also for those who are granted a credit card, the model in incorrect more than 22% of the times for gender 0 and 14% of the time for gender 1.

This is also shown breaking down the confusion matrix by gender and expressing the findings in percentages.


```python
nrf=rf.iloc[:,0:8].copy()
nrf['samples']=nrf.tn+nrf.fn+nrf.fp+nrf.tp
nrf.tp/=nrf.samples/100.
nrf.tn/=nrf.samples/100.
nrf.fp/=nrf.samples/100.
nrf.fn/=nrf.samples/100.

```


```python
pd.options.display.float_format = "{:,.2f}".format
nrf.iloc[:,0:5]
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>xgb all</td>
      <td>43.33</td>
      <td>10.00</td>
      <td>5.83</td>
      <td>40.83</td>
    </tr>
    <tr>
      <th>1</th>
      <td>g=0</td>
      <td>45.24</td>
      <td>10.71</td>
      <td>7.14</td>
      <td>36.90</td>
    </tr>
    <tr>
      <th>2</th>
      <td>g=1</td>
      <td>38.89</td>
      <td>8.33</td>
      <td>2.78</td>
      <td>50.00</td>
    </tr>
  </tbody>
</table>
</div>




```python
from sklearn.metrics import confusion_matrix, fbeta_score, make_scorer, average_precision_score, auc, \
    accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, classification_report, \
    brier_score_loss, precision_recall_curve, roc_curve
```


```python
def get_index_value(sf,col,value):
    """
    returns the DataFrame index for x_test and the array index for y_score
    """
    ind_df=sf[sf[col]==value].index.tolist()
    ind_arr=np.nonzero(sf[col].values==value)[0]
    return ind_df,ind_arr

def get_scores(y_test, y_score, threshold):
    """
    calculates several scores for the classifier
    """
    
    
    y_pred= y_score>= threshold

    cm=confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp =cm.ravel()
#     FAR=fn/(fn+tp)
    FAR=fp/(fp+tn)
    recall=tp/(tp+fn)
    precision=tp/(tp+fp)
    accuracy=(tp+tn)/(tn+ fp+ fn+ tp)
    return [tn, fp, fn, tp,FAR, recall,precision,accuracy]
    
```


```python

ind_df0,ind_a0= get_index_value(X_test,'Gender',0)
ind_df1,ind_a1= get_index_value(X_test,'Gender',1)


```


```python
def plot_roc_mat_subgroup(X_test,y_scoreA,y_testA, col,values,threshold=None):

    for v in values:
        
        ind_df,ind_a= get_index_value(X_test,col,v)
        name='{}= {}'.format(col,v) 
        y_test=y_testA[ind_df]
        y_score=y_scoreA[ind_a]
        
        fpr, tpr, thresholds = roc_curve(y_test,  y_score)
        roc_auc = auc(fpr, tpr)
        hh,=plt.plot(fpr, tpr, lw=1, alpha=1, label='%s (AUC = %0.2f)' % (name,roc_auc));

        if threshold is not None:
            tn, fp, fn, tp,FAR, recall,precision,accuracy=get_scores(y_test , y_score , threshold)
            plt.plot(fp/(fp+tn),recall, 'o',color=hh.get_color())
        
        
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
                   label='Chance', alpha=.8)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('FPR = FP/(FP+TN)')
    plt.ylabel('TPR = Recall = (TP/(TP+FN)) ')
    plt.title('ROC viewed')
    plt.legend(loc="lower right");
    plt.axis('square');
    

```

### ROC curves

```python
plt.figure(figsize=(6,6))
threshold=0.5
plot_roc_mat_subgroup(X_test,y_score,y_test, 'Gender',[0,1],threshold=threshold)
plt.title('ROC by gender for threshold {}'.format(threshold));
```


{% raw %}![alt](/assets/UCI_Credit_Cards_fairness/output_41_0.png){% endraw %}


An additional way of displaying the situation is plotting the individual ROC curves for each gender class, and the position corresponding to a threshold (in this case 0.5).



```python
def calc_f_r(X_test, y_test, y_score):
    thresholds=np.linspace(0,1,500)
    
    
    ind_df0,ind_a0= get_index_value(X_test,'Gender',0)
    ind_df1,ind_a1= get_index_value(X_test,'Gender',1)
    
    r0=[]
    r1=[]
    f0=[]
    f1=[]

    for t in thresholds:
        tn0, fp0, fn0, tp0,FAR0, recall0,precision0,accuracy0 =  get_scores(y_test[ind_df0], y_score[ind_a0], t)
        tn1, fp1, fn1, tp1,FAR1, recall1,precision1,accuracy1 =  get_scores(y_test[ind_df1], y_score[ind_a1], t)

        r0.append(recall0)
        r1.append(recall1)

        f0.append(FAR0)
        f1.append(FAR1)

    r0=np.array(r0)
    r1=np.array(r1)
    f0=np.array(f0)
    f1=np.array(f1)  
    return r0,r1,f0,f1,thresholds
```


### Choice of threshold

```python
# train
r0,r1,f0,f1,thresholds =  calc_f_r(X_train, y_train, y_score_train)

min_t=0
max_t=1


plt.figure(figsize=(12,8))
plt.subplot(221)
plt.plot(thresholds,f0-f1)
plt.xlabel('threshold')
plt.ylabel(r'$\Delta FAR $')
plt.xlim(min_t,max_t);

plt.subplot(222)
plt.plot(thresholds,r0-r1)
plt.xlabel('threshold')
plt.ylabel(r'$\Delta Recall $')
plt.xlim(min_t,max_t);

plt.subplot(223)
plt.plot(thresholds,f0, label='FAR g=0')
plt.plot(thresholds,f1, label='FAR g=1')
plt.legend(loc=1)
plt.xlabel('threshold')
plt.ylabel(r'FAR')
plt.xlim(min_t,max_t);


plt.subplot(224)
plt.plot(thresholds,r0, label='Recall g=0')
plt.plot(thresholds,r1, label='Recall g=1')
plt.legend(loc=3)
plt.xlabel('threshold')
plt.ylabel(r'Recall')
plt.xlim(min_t,max_t);



plt.suptitle('Difference in FAR and Recall depending on the threshold(train)');
```

    invalid value encountered in longlong_scalars
    


{% raw %}![alt](/assets/UCI_Credit_Cards_fairness/output_44_1.png){% endraw %}



One way to enforce fairness is to choose a threshold that results in very similar False Alarm Ratio (FAR).
The plots above suggests to consider thresholds in the interval [0,0.25]


```python
plt.figure(figsize=(6,6))
threshold=0.25
plot_roc_mat_subgroup(X_test,y_score,y_test, 'Gender',[0,1],threshold=threshold)
plt.title('ROC by gender for threshold {}'.format(threshold));
```


{% raw %}![alt](/assets/UCI_Credit_Cards_fairness/output_47_0.png){% endraw %}

## Verification of correction of bias

```python
def metrics_by_gender(rf,X_test,y_test,y_score,threshold, prefix=None):

    col='Gender'
    values=[0,1]

    for v in values:

        ind_df,ind_a= get_index_value(X_test,col,v)
        name='{}= {}'.format(col,v) 
        y_test_g=y_test[ind_df]
        y_score_g=y_score[ind_a]

        y_pred_g= y_score_g> threshold
        if prefix is None:
            rf=dfh.add_metrics(y_test_g,y_score_g,y_pred_g,'g={} t={}'.format(v,threshold),rf )
        else:
            rf=dfh.add_metrics(y_test_g,y_score_g,y_pred_g,'{} g={} t={}'.format(prefix,v,threshold),rf )
        
    rf['FAR']=rf.fp/(rf.fp+rf.tn)
    
    nrf=rf.iloc[:,[0,1,2,3,4,5,6,7,11]].copy()
    nrf['samples']=nrf.tn+nrf.fn+nrf.fp+nrf.tp
    nrf.tp/=nrf.samples/100.
    nrf.tn/=nrf.samples/100.
    nrf.fp/=nrf.samples/100.
    nrf.fn/=nrf.samples/100.

    return (rf,nrf) 

```


```python
threshold=0.25
rf ,nrf= metrics_by_gender(rf,X_test,y_test,y_score,threshold)

```


```python
rf.iloc[:,[0,1,2,3,4,5,6,7,-1]]
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
      <th>FAR</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>xgb all</td>
      <td>52</td>
      <td>12</td>
      <td>7</td>
      <td>49</td>
      <td>0.80</td>
      <td>0.88</td>
      <td>0.91</td>
      <td>0.19</td>
    </tr>
    <tr>
      <th>1</th>
      <td>g=0</td>
      <td>38</td>
      <td>9</td>
      <td>6</td>
      <td>31</td>
      <td>0.78</td>
      <td>0.84</td>
      <td>0.89</td>
      <td>0.19</td>
    </tr>
    <tr>
      <th>2</th>
      <td>g=1</td>
      <td>14</td>
      <td>3</td>
      <td>1</td>
      <td>18</td>
      <td>0.86</td>
      <td>0.95</td>
      <td>0.95</td>
      <td>0.18</td>
    </tr>
    <tr>
      <th>3</th>
      <td>g=0 t=0.25</td>
      <td>33</td>
      <td>14</td>
      <td>4</td>
      <td>33</td>
      <td>0.70</td>
      <td>0.89</td>
      <td>0.89</td>
      <td>0.30</td>
    </tr>
    <tr>
      <th>4</th>
      <td>g=1 t=0.25</td>
      <td>12</td>
      <td>5</td>
      <td>1</td>
      <td>18</td>
      <td>0.78</td>
      <td>0.95</td>
      <td>0.95</td>
      <td>0.29</td>
    </tr>
  </tbody>
</table>
</div>




```python
nrf
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
      <th>FAR</th>
      <th>samples</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>xgb all</td>
      <td>43.33</td>
      <td>10.00</td>
      <td>5.83</td>
      <td>40.83</td>
      <td>0.80</td>
      <td>0.88</td>
      <td>0.91</td>
      <td>0.19</td>
      <td>120</td>
    </tr>
    <tr>
      <th>1</th>
      <td>g=0</td>
      <td>45.24</td>
      <td>10.71</td>
      <td>7.14</td>
      <td>36.90</td>
      <td>0.78</td>
      <td>0.84</td>
      <td>0.89</td>
      <td>0.19</td>
      <td>84</td>
    </tr>
    <tr>
      <th>2</th>
      <td>g=1</td>
      <td>38.89</td>
      <td>8.33</td>
      <td>2.78</td>
      <td>50.00</td>
      <td>0.86</td>
      <td>0.95</td>
      <td>0.95</td>
      <td>0.18</td>
      <td>36</td>
    </tr>
    <tr>
      <th>3</th>
      <td>g=0 t=0.25</td>
      <td>39.29</td>
      <td>16.67</td>
      <td>4.76</td>
      <td>39.29</td>
      <td>0.70</td>
      <td>0.89</td>
      <td>0.89</td>
      <td>0.30</td>
      <td>84</td>
    </tr>
    <tr>
      <th>4</th>
      <td>g=1 t=0.25</td>
      <td>33.33</td>
      <td>13.89</td>
      <td>2.78</td>
      <td>50.00</td>
      <td>0.78</td>
      <td>0.95</td>
      <td>0.95</td>
      <td>0.29</td>
      <td>36</td>
    </tr>
  </tbody>
</table>
</div>



As discussed, the new threshold 0.25 results in a less unfair classification.


```python
# Does this still hold for the smaller validation set (50 data points)?
y_pred2 = xg_cl.predict(X_val)
y_score2 = xg_cl.predict_proba(X_val)[:,1]

rf ,nrf= metrics_by_gender(rf,X_val,y_val,y_score2,threshold, prefix='val')
```


```python
rf.iloc[:,[0,1,2,3,4,5,6,7,-1]]
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
      <th>FAR</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>xgb all</td>
      <td>52</td>
      <td>12</td>
      <td>7</td>
      <td>49</td>
      <td>0.80</td>
      <td>0.88</td>
      <td>0.91</td>
      <td>0.19</td>
    </tr>
    <tr>
      <th>1</th>
      <td>g=0</td>
      <td>38</td>
      <td>9</td>
      <td>6</td>
      <td>31</td>
      <td>0.78</td>
      <td>0.84</td>
      <td>0.89</td>
      <td>0.19</td>
    </tr>
    <tr>
      <th>2</th>
      <td>g=1</td>
      <td>14</td>
      <td>3</td>
      <td>1</td>
      <td>18</td>
      <td>0.86</td>
      <td>0.95</td>
      <td>0.95</td>
      <td>0.18</td>
    </tr>
    <tr>
      <th>3</th>
      <td>g=0 t=0.25</td>
      <td>33</td>
      <td>14</td>
      <td>4</td>
      <td>33</td>
      <td>0.70</td>
      <td>0.89</td>
      <td>0.89</td>
      <td>0.30</td>
    </tr>
    <tr>
      <th>4</th>
      <td>g=1 t=0.25</td>
      <td>12</td>
      <td>5</td>
      <td>1</td>
      <td>18</td>
      <td>0.78</td>
      <td>0.95</td>
      <td>0.95</td>
      <td>0.29</td>
    </tr>
    <tr>
      <th>5</th>
      <td>val g=0 t=0.25</td>
      <td>46</td>
      <td>21</td>
      <td>4</td>
      <td>49</td>
      <td>0.70</td>
      <td>0.92</td>
      <td>0.84</td>
      <td>0.31</td>
    </tr>
    <tr>
      <th>6</th>
      <td>val g=1 t=0.25</td>
      <td>23</td>
      <td>6</td>
      <td>1</td>
      <td>20</td>
      <td>0.77</td>
      <td>0.95</td>
      <td>0.83</td>
      <td>0.21</td>
    </tr>
  </tbody>
</table>
</div>




```python
nrf
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
      <th>FAR</th>
      <th>samples</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>xgb all</td>
      <td>43.33</td>
      <td>10.00</td>
      <td>5.83</td>
      <td>40.83</td>
      <td>0.80</td>
      <td>0.88</td>
      <td>0.91</td>
      <td>0.19</td>
      <td>120</td>
    </tr>
    <tr>
      <th>1</th>
      <td>g=0</td>
      <td>45.24</td>
      <td>10.71</td>
      <td>7.14</td>
      <td>36.90</td>
      <td>0.78</td>
      <td>0.84</td>
      <td>0.89</td>
      <td>0.19</td>
      <td>84</td>
    </tr>
    <tr>
      <th>2</th>
      <td>g=1</td>
      <td>38.89</td>
      <td>8.33</td>
      <td>2.78</td>
      <td>50.00</td>
      <td>0.86</td>
      <td>0.95</td>
      <td>0.95</td>
      <td>0.18</td>
      <td>36</td>
    </tr>
    <tr>
      <th>3</th>
      <td>g=0 t=0.25</td>
      <td>39.29</td>
      <td>16.67</td>
      <td>4.76</td>
      <td>39.29</td>
      <td>0.70</td>
      <td>0.89</td>
      <td>0.89</td>
      <td>0.30</td>
      <td>84</td>
    </tr>
    <tr>
      <th>4</th>
      <td>g=1 t=0.25</td>
      <td>33.33</td>
      <td>13.89</td>
      <td>2.78</td>
      <td>50.00</td>
      <td>0.78</td>
      <td>0.95</td>
      <td>0.95</td>
      <td>0.29</td>
      <td>36</td>
    </tr>
    <tr>
      <th>5</th>
      <td>val g=0 t=0.25</td>
      <td>38.33</td>
      <td>17.50</td>
      <td>3.33</td>
      <td>40.83</td>
      <td>0.70</td>
      <td>0.92</td>
      <td>0.84</td>
      <td>0.31</td>
      <td>120</td>
    </tr>
    <tr>
      <th>6</th>
      <td>val g=1 t=0.25</td>
      <td>46.00</td>
      <td>12.00</td>
      <td>2.00</td>
      <td>40.00</td>
      <td>0.77</td>
      <td>0.95</td>
      <td>0.83</td>
      <td>0.21</td>
      <td>50</td>
    </tr>
  </tbody>
</table>
</div>



# Conclusion

Despite the very small dataset it's possible to partially address the unfairness for the gender feature.
The choice of threshold is confirmed by the validation set.

