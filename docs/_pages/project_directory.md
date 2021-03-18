---
title: "Projects"
permalink: /projects/
excerpt: "Project portfolio"
last_modified_at: 2019-04-18T15:53:52-04:00
toc: true
---

Dealing with the complexity of real-life projects is essential to really grow as an expert in Data Science and Machine Learning.
There are many excellent online courses for those who want to improve their Data Science skills but nothing beats practice.
Unfortunately, I cannot write about my projects at work, except for academic publications.


So I am starting a directory about projects of interest from public sources such as the UCI machine learning repository or Kaggle.
Some projects are from the excellent course on Udacity. I was impressed with the complexity of the projects and with throughness of the expert review.


## [Credit card frauds](/credit_card_fraud/)

The dataset is extremely unbalanced (0.172% positive class), which originates a number of issues. 

Challenges:
* extremely unbalanced dataset

Techniques used:
* Feature selection
* Gradient Boosted Trees


## [Credit card applications](/credit_card/)

Gradient boosted trees (XGBoost) are used to classify credit card applications.
The dataset is extremely tiny (690 examples), which originates a number of issues. 

Challenges:
* very small dataset

Techniques used:
* Learning Curves
* Gradient Boosted Trees
* Fairness analysis

## [Air Quality Sensor Prediction](/gas_sensor/)

Neural networks (LSTM) are used to predict the benzene gas concentration using time-series data from an integrated sensor device.

Challenges:
* analysis of time series

Techniques used:
* LSTM


## [Sparkify User Churn](/songs/)

The dataset containing simulated user behavior for a fictitional music streaming service is extremely large (12 GB). 
Dealing efficiently with such large datasets is challenging.

Challenges:
* extremely large dataset

Techniques used:
* SQL
* Spark

## [Starbucks Capstone (Udacity)](/starbucks/)

Large project aimed at predicting Starbucks' customer order patterns and sensitivity to promotional offers.
As in many real-life projects, the critical part of this project is understanding the project and the data.

Challenges:
* the offer periods overlap strongly and it's not straightforward to cleanly separate train and test sets. Obtaining several non-overlapping cross-validation folds is challenging.

Techniques used:
* use of SQL to efficiently merge and combine data sets based on time intervals 
* Custom visualization of time series based on spectrograms


## [Twitter Disaster Detection (Udacity)](/twitter/)
NLP analysis of twitter messages from disaster areas with both BOW and NN models. This is a smaller project. 

Challenges:
* highly imbalanced classes 

Features:
* Natural language processing

## [Arvato Capstone](/arvato)
Supervised Learning model to predict which customers will respond to an advertising campaign.

Challenges:
* highly imbalanced classes 
* imputing of missing values

