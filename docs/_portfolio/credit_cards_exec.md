---
  
title: "Credit Card Application"
permalink: /credit_card/
excerpt: "Prediction of the outcome of credit card applications using a tiny dataset"
header:
  image: /assets/UCI_Credit_Cards/card_splash1.jpg
  teaser: assets/UCI_Credit_Cards/card_splash1.jpg 
last_modified_at: 2020-06-11T15:53:52-04:00
toc: false
toc_label: "Page Index"
author_profile: true
classes: wide
---




## Overview
The dataset comprises decisions on credit card applications based on the features of the applicant (ZipCode, employment status, education...).
The dataset comprises only 690 data points.

## Goal
The use case is to predict the outcome of a credit card application, highlighting issues in dealing with tiny datasets.
Finally, the models obtained are checked for fairness.


## Analysis

### [Goal and EDA](/credit_card/cc_eda/)
Problem statement and initial data exploration.

### [Classification Model](/credit_card/cc_model/)
Simple approach based on gradient boosted tree classifier.

### [Fairness analysis](/credit_card/cc_fairness/)
Analysis and improvement of fairness of the classification.


#### Tools
- XGBoost
- scikit-learn
- SHAP
- seaborn

#### Techniques
- learning curve
- model stability
- hyperparameter tuning
- fairness analysis