---
  
title: "Credit Card Fraud Detection"
permalink: /credit_card_fraud/
excerpt: "Detection of credit card fraudolent transactions"
header:
  image: /assets/CC_fraud/credit_fraud.jpg
  teaser: assets/CC_fraud/credit_fraud.jpg 
last_modified_at: 2020-06-11T15:53:52-04:00
toc: false
toc_label: "Page Index"
author_profile: true
classes: wide
---



## Overview
This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. 
The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions.
It contains only numerical input variables which are the result of a PCA transformation. 

## Goal
The goal is to train a classifier able to distinguish fraudolent transactions (obviously a very small percentage of total transactions).

## Analysis

### [Goal and EDA](/credit_card_fraud/ccf_model/)
Problem statement and initial data exploration. Several feature selection techniques are used to identify promising models


#### Tools
- XGBoost
- scikit-learn

#### Techniques
- learning curve
- feature selection

