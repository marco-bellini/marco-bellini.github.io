---
  
title: "TW3: Hyperparameter Tuning"
permalink: /twitter/hyperparameter/
excerpt: "Twitter Response: Classifier Hyperparameter Tuning"
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

### 2.2.2 Hyperparameter tuning 

The models will be limited to:
* Logistic regression with strong regularization (L1)
* Random forests with limited depth and number of features

SVC are excluded because the computation costs increase quadratically with the dimensionality of the problem.



```python
from sklearn.model_selection import validation_curve

clf_lr=LogisticRegression(random_state=0, solver='liblinear',penalty='l1',max_iter=200,class_weight='balanced' )

```


```python

def plot_validation_curve(clf_lr,bow1k_bal,param_name,param_range,scoring="f1",xlog=True):

    train_scores, test_scores = validation_curve(
        clf_lr, bow1k_bal['Xtrain'], bow1k_bal['Ytrain'], param_name=param_name, param_range=param_range,
        cv=5, scoring=scoring, n_jobs=4)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    plt.ylabel("Score")
    plt.ylim(0.0, 1.1)
    lw = 2
    if xlog:
        plt.semilogx(param_range, train_scores_mean, label="Training score",
                 color="darkorange", lw=lw)
    else:
        plt.plot(param_range, train_scores_mean, label="Training score",
                 color="darkorange", lw=lw)
    
    plt.fill_between(param_range, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.2,
                     color="darkorange", lw=lw)
    if xlog:
        plt.semilogx(param_range, test_scores_mean, label="Cross-validation score",
                 color="navy", lw=lw)
    else:
        plt.plot(param_range, test_scores_mean, label="Cross-validation score",
                 color="navy", lw=lw)
    plt.fill_between(param_range, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.2,
                     color="navy", lw=lw)
    
    plt.legend(loc="best")

```


```python
param_range=np.logspace(-3,0,10)
plot_validation_curve(clf_lr,bow1k_bal,"C",param_range,scoring="f1")

plt.title("Validation Curve with LR")
plt.xlabel("C");

```


{% raw %}![alt](/assets/nlp_disaster/output_87_1.png){% endraw %}





```python
clf= RandomForestClassifier(  criterion="entropy",class_weight="balanced")

param_range=[5,10,20,50,100,200]
plot_validation_curve(clf,bow1k_bal,"n_estimators",param_range,scoring="f1")

plt.title("Validation Curve with RF")
plt.xlabel("N_est");
```

{% raw %}![alt](/assets/nlp_disaster/output_89_0.png){% endraw %}

{% raw %}![alt](/assets/nlp_disaster/output_90_0.png){% endraw %}

{% raw %}![alt](/assets/nlp_disaster/output_91_0.png){% endraw %}

{% raw %}![alt](/assets/nlp_disaster/output_92_0.png){% endraw %}


