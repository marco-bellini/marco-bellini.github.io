---
  
title: "LSTM for prediction of Gas Sensor signal"
permalink: /gas_sensor/
excerpt: "Neural networks are used to perform sensor fusion on air quality timeseries."
header:
  image: /assets/ibm_gas_sensor/gas_sensor.jpg
  teaser: assets/ibm_gas_sensor/gas_sensor.jpg 

last_modified_at: 2020-06-11T15:53:52-04:00
toc: false
toc_label: "Page Index"
author_profile: true
classes: wide
---


## Overview
The dataset comprises time series from different gas sensors and it can be used to study sensors fusion topics.

## Goal
The use case is realizing a machine learning algorithm able to infer accurate air quality information (Benzene concentration) from data collected by an array of 5 metal oxide chemical sensors embedded in an Air Quality Chemical Multisensor Device. The ground truth data is obtained reference certified analyzer. 
I plan to extend the study when I have time to address detection of faulty sensors and mitigation of missing input signals through sensor fusion techniques.


## Analysis

### [Goal and EDA](/gas_sensor/gas_eda/)
Problem statement and initial data exploration.

### [Basic Neural Network Model](/gas_sensor/gas_model/)
Simple baseline models and first neural network approach.


#### Tools
- Tensorflow
- statmodels

#### Techniques
- LSTM
- Time Series
- Deep Learning

This project was used as the capstone project for the IBM Advanced Data Science Specialization Certificate.
