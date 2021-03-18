---
title: "CC 1: Goal and EDA"
permalink: /credit_card/cc_eda/
excerpt: "Credit Cards: Goal and EDA"
last_modified_at: 2020-06-11T15:53:52-04:00
toc: true
toc_label: "Page Index"
toc_sticky: true
author_profile: true
classes: post
tags:
  - EDA
  - Visualization
categories:
  - Credit Card
---



# Credit Card Approval

The UCI Credit Card Approval DataBase is available [here](https://archive.ics.uci.edu/ml/datasets/credit+approval) at the [UCI Repository](https://archive.ics.uci.edu/ml/index.php).

This file concerns credit card applications. All attribute names and values have been changed to meaningless symbols to protect confidentiality of the data.
It features 360 examples and 15 features and two classes (Approved or Not Approved).

This dataset is interesting because there is a good mix of attributes -- continuous, nominal with small numbers of values, and nominal with larger numbers of values. There are also a few missing values.


The dataset is interesting because there are very few data points.
It is essential to be very careful about using data to avoid leaking.


# Approach

When developing an algorithm for credit card applications it's important to consider the following:
* what kind of output should the model give (approved / rejected, confidence to approve to be considered by a rule-based system)?
* how was the input data collected? Is it possible that the ground truth contains errors?
* what performance targets should the algorithm be optimized for (high precision, high recall, ...)?
* what is the consequence of approving or rejecting incorrectly a potential customer?

It's important to remark that this database is incredibly small, especially considering the application.
Dealing with small datasets raises a number of challenges, which will be discussed here.
Since the dataset is very small it's not split prior to the EDA phase. This is a potential concern.


The following steps are taken:
* explore the dataset: quantile comparisons and paired histograms suggest that some features are more relevant than others
* feature selection for choosing the candidate features
* learning curve analysis is used to estimate a split of the dataset
* logistic regression, random forests and gradient boosted trees are explored as viable algorithms. Binning and log transformations are applied.
* hyperparameter tuning
* fairness analysis: gender fairness of the model is assessed. The model is further tuned to improve fairness

Next steps:
* error analysis



## Data import

The data is provided as CSV file, including a mapping for feature names and an initial categorization into numerical, binary and categorical features.


```python
# we restore the original feature names, along with initial feature classification
feature_names=['A%d' % d for d in range(1,17)]
feat_cont=['A2','A3','A8','A11','A14','A15']
feat_bin=['A1','A9','A10','A12','A16']
feat_cat=list( set( feature_names ) -set(feat_cont) -set(feat_bin) )

real_names=['Gender','Age','Debt','Married','BankCustomer','EducationLevel','Ethnicity','YearsEmployed','PriorDefault','Employed',
            'CreditScore','DriversLicense','Citizen','ZipCode','Income','ApprovalStatus']

rename_dict=dict( zip(feature_names,real_names) )

# online
# df = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/credit-screening/crx.data',sep=',',
#                  names=real_names, dtype=str,na_values='?')

# offline
df = pd.read_csv('crx.data',sep=',',
                 names=real_names, dtype=str,na_values='?')


feat_cont=[rename_dict[fc] for fc in feat_cont]
feat_bin=[rename_dict[fc] for fc in feat_bin]
feat_cat=[rename_dict[fc] for fc in feat_cat]

for col in feat_cont:
  df[col]=df[col].astype(float)

df['Gender']=df['Gender']=='a'
df['ApprovalStatus']=df['ApprovalStatus']=='+'

for col in ['PriorDefault','Employed','DriversLicense']:
  df[col]=df[col]=='t'

# we copy the original database
cf=df.copy()
 


```

## Exploratory Data Analysis




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 690 entries, 0 to 689
    Data columns (total 16 columns):
     #   Column          Non-Null Count  Dtype  
    ---  ------          --------------  -----  
     0   Gender          690 non-null    bool   
     1   Age             678 non-null    float64
     2   Debt            690 non-null    float64
     3   Married         684 non-null    object 
     4   BankCustomer    684 non-null    object 
     5   EducationLevel  681 non-null    object 
     6   Ethnicity       681 non-null    object 
     7   YearsEmployed   690 non-null    float64
     8   PriorDefault    690 non-null    bool   
     9   Employed        690 non-null    bool   
     10  CreditScore     690 non-null    float64
     11  DriversLicense  690 non-null    bool   
     12  Citizen         690 non-null    object 
     13  ZipCode         677 non-null    float64
     14  Income          690 non-null    float64
     15  ApprovalStatus  690 non-null    bool   
    dtypes: bool(5), float64(6), object(5)
    memory usage: 62.8+ KB
    


```python
df.describe()
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
      <th>Age</th>
      <th>Debt</th>
      <th>YearsEmployed</th>
      <th>CreditScore</th>
      <th>ZipCode</th>
      <th>Income</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>678.000000</td>
      <td>690.000000</td>
      <td>690.000000</td>
      <td>690.00000</td>
      <td>677.000000</td>
      <td>690.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>31.568171</td>
      <td>4.758725</td>
      <td>2.223406</td>
      <td>2.40000</td>
      <td>184.014771</td>
      <td>1017.385507</td>
    </tr>
    <tr>
      <th>std</th>
      <td>11.957862</td>
      <td>4.978163</td>
      <td>3.346513</td>
      <td>4.86294</td>
      <td>173.806768</td>
      <td>5210.102598</td>
    </tr>
    <tr>
      <th>min</th>
      <td>13.750000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>22.602500</td>
      <td>1.000000</td>
      <td>0.165000</td>
      <td>0.00000</td>
      <td>75.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>28.460000</td>
      <td>2.750000</td>
      <td>1.000000</td>
      <td>0.00000</td>
      <td>160.000000</td>
      <td>5.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>38.230000</td>
      <td>7.207500</td>
      <td>2.625000</td>
      <td>3.00000</td>
      <td>276.000000</td>
      <td>395.500000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>80.250000</td>
      <td>28.000000</td>
      <td>28.500000</td>
      <td>67.00000</td>
      <td>2000.000000</td>
      <td>100000.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# we notice that ZipCode is considered numerical instead of categorical
len(np.unique(df.ZipCode))
```




    183




```python
df[feat_cat+feat_bin].describe()
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
      <th>Ethnicity</th>
      <th>Citizen</th>
      <th>Married</th>
      <th>EducationLevel</th>
      <th>BankCustomer</th>
      <th>Gender</th>
      <th>PriorDefault</th>
      <th>Employed</th>
      <th>DriversLicense</th>
      <th>ApprovalStatus</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>681</td>
      <td>690</td>
      <td>684</td>
      <td>681</td>
      <td>684</td>
      <td>690</td>
      <td>690</td>
      <td>690</td>
      <td>690</td>
      <td>690</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>9</td>
      <td>3</td>
      <td>3</td>
      <td>14</td>
      <td>3</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>top</th>
      <td>v</td>
      <td>g</td>
      <td>u</td>
      <td>c</td>
      <td>g</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>399</td>
      <td>625</td>
      <td>519</td>
      <td>137</td>
      <td>519</td>
      <td>480</td>
      <td>361</td>
      <td>395</td>
      <td>374</td>
      <td>383</td>
    </tr>
  </tbody>
</table>
</div>




```python
df['ApprovalStatus'].value_counts(normalize=True)
```




    False    0.555072
    True     0.444928
    Name: ApprovalStatus, dtype: float64



The classes are reasonably well balanced.

### Missing Values

The dataset is tiny: only 690 records are available. 
However the dataset is reasonably balanced.
We will now look for missing values and outliers.



```python
n_records=df.shape[0]

print('shape:',df.shape)
print()
print('number of missing values:')
for col in df.columns:
  nna=n_records-df[col].dropna().shape[0]
  if nna>0:
    print('{col:15}:{nna}'.format(col=col,nna=nna))
```

    shape: (690, 16)
    
    number of missing values:
    Age            :12
    Married        :6
    BankCustomer   :6
    EducationLevel :9
    Ethnicity      :9
    ZipCode        :13
    




{% raw %}![alt](/assets/UCI_Credit_Cards/output_16_0.png){% endraw %}


{% raw %}![alt](/assets/UCI_Credit_Cards/output_17_0.png){% endraw %}


Only less than 1% of the rows have a high number of missing values. We can discard these rows or use imputation.


## Numerical variables


```python
feat_cont
```




['Age', 'Debt', 'YearsEmployed', 'CreditScore', 'ZipCode', 'Income']



### Correlation


```python
dfh.plot_correlation_matrix(cf[['Age', 'Debt', 'YearsEmployed', 'CreditScore', 'Income','ApprovalStatus']],figsize=(10,10))
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
      <th>Age</th>
      <th>Debt</th>
      <th>YearsEmployed</th>
      <th>CreditScore</th>
      <th>Income</th>
      <th>ApprovalStatus</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Age</th>
      <td>1.000000</td>
      <td>0.202317</td>
      <td>0.395751</td>
      <td>0.185912</td>
      <td>0.018553</td>
      <td>0.162881</td>
    </tr>
    <tr>
      <th>Debt</th>
      <td>0.202317</td>
      <td>1.000000</td>
      <td>0.298902</td>
      <td>0.271207</td>
      <td>0.123121</td>
      <td>0.206294</td>
    </tr>
    <tr>
      <th>YearsEmployed</th>
      <td>0.395751</td>
      <td>0.298902</td>
      <td>1.000000</td>
      <td>0.322330</td>
      <td>0.051345</td>
      <td>0.322475</td>
    </tr>
    <tr>
      <th>CreditScore</th>
      <td>0.185912</td>
      <td>0.271207</td>
      <td>0.322330</td>
      <td>1.000000</td>
      <td>0.063692</td>
      <td>0.406410</td>
    </tr>
    <tr>
      <th>Income</th>
      <td>0.018553</td>
      <td>0.123121</td>
      <td>0.051345</td>
      <td>0.063692</td>
      <td>1.000000</td>
      <td>0.175657</td>
    </tr>
    <tr>
      <th>ApprovalStatus</th>
      <td>0.162881</td>
      <td>0.206294</td>
      <td>0.322475</td>
      <td>0.406410</td>
      <td>0.175657</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



{% raw %}![alt](/assets/UCI_Credit_Cards/output_28_2.png){% endraw %}





### Pairwise plots

Pairwise plot from seaborn are useful to discover relationships between features.


```python
g = sns.PairGrid(df.dropna(), vars=['Age', 'Debt', 'YearsEmployed', 'CreditScore', 'Income'], hue="ApprovalStatus")
g.map_diag(plt.hist,alpha=0.5)
g.map_offdiag(plt.scatter,alpha=0.5)
g.add_legend();
```

{% raw %}![alt](/assets/UCI_Credit_Cards/output_31_0.png){% endraw %}



We notice that several variables are clustered around zero but exhibit also large values.
A logarithmic transformation is useful to rescale the data.


```python
df['Debt_log']=np.log10(df['Debt']+1)
df['Income_log']=np.log10(df['Income']+1)
df['CreditScore_log']=np.log10(df['CreditScore']+1)

```


```python
g = sns.PairGrid(df.dropna(), vars=['Age', 'Debt_log', 'YearsEmployed', 'CreditScore_log', 'Income_log'], hue="ApprovalStatus")
g.map_diag(plt.hist,alpha=0.5)
g.map_offdiag(plt.scatter,alpha=0.5)
g.add_legend();
```

{% raw %}![alt](/assets/UCI_Credit_Cards/output_34_0.png){% endraw %}



The logarithmic transformation suggests some structure. 
Binning could be applied to introduce some regularization.

### Outliers

With only 690 data points it's difficult to use algorithms such as IsolationForest.
We can use a boxplot for a first identification of outliers. Given the range of some variables, we also apply a log transform.


```python
ycol='ApprovalStatus'
c=0

plt.figure(figsize=(18,12))
for xcol in num_feat+['Income_log','CreditScore_log']:
    plt.subplot(3,3,c+1)
    sns.boxplot(data=df,y=xcol,x=ycol);
    c+=1
```


{% raw %}![alt](/assets/UCI_Credit_Cards/output_40_0.png){% endraw %}


However it is valuable is to look at the extreme quantiles of the distribution.



```python
num_feat=['Age','Debt','YearsEmployed','CreditScore','Income']
qf=df[num_feat].quantile([.1,.95])
```


```python
low_q=pd.DataFrame()
for col in qf.columns:
    new=df[(df[col]<qf.loc[0.1][col])]['ApprovalStatus'].value_counts().to_frame(name=col)
    low_q=pd.concat([low_q, new] , axis=1 ).fillna(0)
low_q
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
      <th>Age</th>
      <th>Debt</th>
      <th>YearsEmployed</th>
      <th>CreditScore</th>
      <th>Income</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>False</th>
      <td>45</td>
      <td>37</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>True</th>
      <td>18</td>
      <td>27</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



The low outlier values of `Age` seems to have strong predictive power: young people seem to be denied more frequently.


```python
hi_q=pd.DataFrame()
for col in qf.columns:
    new=df[(df[col]<qf.loc[0.95][col])]['ApprovalStatus'].value_counts().to_frame(name=col)
    hi_q=pd.concat([hi_q, new] , axis=1 ).fillna(0)
hi_q
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
      <th>Age</th>
      <th>Debt</th>
      <th>YearsEmployed</th>
      <th>CreditScore</th>
      <th>Income</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>False</th>
      <td>362</td>
      <td>372</td>
      <td>375</td>
      <td>380</td>
      <td>379</td>
    </tr>
    <tr>
      <th>True</th>
      <td>282</td>
      <td>282</td>
      <td>280</td>
      <td>259</td>
      <td>276</td>
    </tr>
  </tbody>
</table>
</div>



The high outlier values of seem to have some predictive power but more analysis is needed.


```python
bins=20
density=True
importlib.reload(dfh)

feat_cont_ext=num_feat 

plt.figure(figsize=(18,8))
for c in range(0,len(feat_cont_ext)):
    plt.subplot(2,3,c+1)
    dfh.compare_histograms(df,feat_cont_ext[c],'ApprovalStatus',density=True,values=[False,True],quantiles=[0.1,0.9]);


```

{% raw %}![alt](/assets/UCI_Credit_Cards/output_37_0.png){% endraw %}




```python
dfh.compare_histograms(df,'Income_log','ApprovalStatus',density=True,values=[False,True]);
```

{% raw %}![alt](/assets/UCI_Credit_Cards/output_38_0.png){% endraw %}




```python
dfh.compare_histograms(df,'CreditScore_log','ApprovalStatus',density=True,values=[False,True]);
```


{% raw %}![alt](/assets/UCI_Credit_Cards/output_39_0.png){% endraw %}





### Low and upper quantiles

These plots are used to check the predictive power of low and upper quantiles.
In anomaly or fault detection low and upper quantiles can show significantly different behavior.

```python
def quantilize(df, features,q_labels=['0-10Q','10-90Q','90-100Q'], q_cuts=[0,0.1,0.9,1] ):
    for feat in features:
        df['{}_q'.format(feat)]=pd.qcut(df[feat],q=q_cuts,labels=q_labels)
    return df

# df=quantilize(df,['Age','Debt','YearsEmployed','CreditScore_log','Income_log'] ,q_labels=['0-10Q','10-90Q','90-100Q'], q_cuts=[0,0.15,0.9,1] )
df=quantilize(df,['Age','Debt',] ,q_labels=['0-10Q','10-90Q','90-100Q'], q_cuts=[0,0.1,0.9,1] )
df=quantilize(df,['YearsEmployed'] ,q_labels=['0-15Q','15-90Q','90-100Q'], q_cuts=[0,0.15,0.9,1] )
df=quantilize(df,['CreditScore_log'] ,q_labels=['0-60Q','60-90Q','90-100Q'], q_cuts=[0,0.6,0.9,1] )
df=quantilize(df,['Income_log'] ,q_labels=['0-60Q','60-90Q','90-100Q'], q_cuts=[0,0.5,0.9,1] )

```

We notice that some features have different distribution for the people who are approved or not.

```python
sns.catplot("Age_q", col="ApprovalStatus",kind="count", data=df.dropna(),height=3);
```

{% raw %}![alt](/assets/UCI_Credit_Cards/output_43_0.png){% endraw %}




```python
sns.catplot("Debt_q", col="ApprovalStatus",kind="count", data=df.dropna(),height=3);
```

{% raw %}![alt](/assets/UCI_Credit_Cards/output_44_0.png){% endraw %}




```python
sns.catplot("YearsEmployed_q", col="ApprovalStatus",kind="count", data=df.dropna(),height=3);
```


{% raw %}![alt](/assets/UCI_Credit_Cards/output_45_0.png){% endraw %}




```python
sns.catplot("CreditScore_log_q", col="ApprovalStatus",kind="count", data=df.dropna(),height=3);
```


{% raw %}![alt](/assets/UCI_Credit_Cards/output_46_0.png){% endraw %}


```python
sns.catplot("Income_log_q", col="ApprovalStatus",kind="count", data=df.dropna(),height=3);
```


{% raw %}![alt](/assets/UCI_Credit_Cards/output_47_0.png){% endraw %}

### Findings for Numerical Variables

* `CreditScore` has the highest correlation with `ApprovalStatus`, especially people with zero credit score are denied.
* `Income`, `CreditScore` and possibly `YearsEmployed` would benefit from a log transformation to compress the range of values, possibly combined with binning.
* the other features seem have predictive value, e.g. people with younger `Age` are more frequently denied

## Categorical and Binary Variables

The quickest way to explore binary and categorical features is to plot the count (or frequency) of the values of a feature for both approved and denied customers.


### ZipCodes
For example we can plot the counts for ZipCodes, using a threshold of 2 counts minimum.

```python
# sns.countplot(data=df,hue='ApprovalStatus',x='ZipCode');
zip_dist=df.groupby(by='ApprovalStatus')['ZipCode'].value_counts()

zip_dist.loc[zip_dist>2].unstack().T.plot(kind='bar',figsize=(8,6))
plt.title('ZipCodes show some trend')
plt.ylabel('count | count >2');

```
{% raw %}![alt](/assets/UCI_Credit_Cards/output_50_0.png){% endraw %}


### Binary Features

In the case of binary features, it's best to present frequency (as a fraction of 1.0):

```python
def bin_proportion(df,xcol,ycol,xvalue=True):
  S=df.groupby(by=ycol)[xcol].value_counts(normalize=True)
  S.name='X'
  S=S.to_frame().reset_index()

  S=S.loc[S[xcol]==xvalue].drop(columns=[xcol])
  S=S.rename(columns={'X':xcol+'=='+str(xvalue)})
  return(S)

```


```python
plt.figure(figsize=(12,8))
for c in range(0,len(feat_bin)-1):
    ax=plt.subplot(2,2,c+1)
    S=df.groupby(by='ApprovalStatus')[feat_bin[c]].value_counts(normalize=True)
    S.unstack().plot(kind='bar',ax=ax);
    plt.ylabel(feat_bin[c])
    plt.legend(loc=3)
```


{% raw %}![alt](/assets/UCI_Credit_Cards/output_52_0.png){% endraw %}

It seems that `Gender` and `DriverLicense` are not relevant, since the proportions are similar regardless of the outcome of the application.

On the other side, `PriorDefault` (true if there was no prior default) and `Employed` are very important predictors. 


### Categorical Features

```python
ycol='ApprovalStatus'

plt.figure(figsize=(18,12))
for c in range(0,len(feat_cat)):
  xcol=feat_cat[c]
  ax=plt.subplot(3,2,c+1)
  
  S=df.groupby(by=ycol)[xcol].value_counts(normalize=True)
  S.name='percentage'
  S=S.to_frame().reset_index()

  sns.barplot(data=S,x=ycol,hue=xcol,y='percentage')
```

### Comparison of histograms


{% raw %}![alt](/assets/UCI_Credit_Cards/output_54_0.png){% endraw %}


```python
dfh.compare_histograms(df,'EducationLevel','ApprovalStatus',density=True,values=[False,True],quantiles=None);
```


{% raw %}![alt](/assets/UCI_Credit_Cards/output_55_0.png){% endraw %}


```python
dfh.compare_histograms(df,'Ethnicity','ApprovalStatus',density=True,values=[False,True],quantiles=None);
```


{% raw %}![alt](/assets/UCI_Credit_Cards/output_56_0.png){% endraw %}

`EducationLevel` seems to have a meaningful effect for some values (`ff`,`i`,`k`,`x`,...).

`Ethnicity` seems to have some limited impact for some values (`ff`, `h`). 




It seems that `EducationLevel` is relevant, but it is not obvious to reach a conclusion for the other features.
It's better to run a multinomial <img src="https://render.githubusercontent.com/render/math?math=\chi^{2}"> independence test for the categorical variables.


```python
pv_cat=dfh.multinomial_chi2_independence(df[feat_cat+feat_bin+['ZipCode']].drop(columns=['ApprovalStatus'])
                                         ,df['ApprovalStatus'],return_series=True)
```


```python
display(pv_cat.sort_values().to_frame(name='p-value'))
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
      <th>p-value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>PriorDefault</th>
      <td>3.118590e-79</td>
    </tr>
    <tr>
      <th>Employed</th>
      <td>5.675727e-33</td>
    </tr>
    <tr>
      <th>EducationLevel</th>
      <td>3.499930e-15</td>
    </tr>
    <tr>
      <th>Ethnicity</th>
      <td>3.625453e-07</td>
    </tr>
    <tr>
      <th>Married</th>
      <td>2.010680e-06</td>
    </tr>
    <tr>
      <th>BankCustomer</th>
      <td>2.010680e-06</td>
    </tr>
    <tr>
      <th>ZipCode</th>
      <td>5.489098e-03</td>
    </tr>
    <tr>
      <th>Citizen</th>
      <td>1.009429e-02</td>
    </tr>
    <tr>
      <th>DriversLicense</th>
      <td>4.509459e-01</td>
    </tr>
    <tr>
      <th>Gender</th>
      <td>4.985346e-01</td>
    </tr>
  </tbody>
</table>
</div>


## Conclusions of EDA

* `PriorDefault` and `Employed` and `CreditScore` seem to be the most important features.
* `Income`, `CreditScore` and possibly `YearsEmployed` would benefit from a log transformation to compress the range of values, possibly combined with binning.
* `Gender` and `DriversLicense` seem to have little importance.

We are going to see if logarithmic transformation and / or binning improve predictiviness.

