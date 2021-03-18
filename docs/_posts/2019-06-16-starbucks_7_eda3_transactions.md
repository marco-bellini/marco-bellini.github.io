---
  
title: "SB 5: Exploratory Data Analysis: Transactions"
permalink: /starbucks/starbucks_eda3_transactions/
excerpt: "Starbucks Case: Exploring the transaction data, use of advanced custom visualization"
last_modified_at: 2019-04-18T15:53:52-04:00
toc: true
toc_label: "Page Index"
toc_sticky: true
author_profile: true
classes: post
tags:
  - EDA
  - Visualization
categories:
  - Starbucks  
---



## Introduction

We explore the dataframe obtained in the [previous session](/starbucks/starbucks_fe/) with a number of statistical and visualization tools. 
The goal is to get more familiar with the data and identify potential issues, new features and relevant business concerns or opportunities.
In this section we focus on the transactions, that were conveniently separated in the dataframes `transactions_during_offer` and `transactions_outside_offer`.

### 4.1) The need for custom visualization

We can quickly plot the transaction during and outside offers with a scatterplot. 
We notice large outliers but we assume that they are correct data relating to large transactions, since the data collection process for transactions is automated.
In a real situation, it would be useful to follow up to understand better such outliers.

{% raw %}![alt](/assets/starbucks/eda_trans_full_dots.png){% endraw %}

We not that scatterplot are not efficient to summarize the data in the range of interest.

{% raw %}![alt](/assets/starbucks/eda_trans_zoom_dots.png){% endraw %}

### 4.3) Histogram plots and periodicity 

To understand better the trends in purchases we plot the histograms of purchases outside of offers for each week of the analyzed period. 

{% raw %}![alt](/assets/starbucks/eda_trans_outside_weekly.png){% endraw %}

We note that the distribution of purchases is likely bimodal. 
It would be interested to separate the distributions at approx. 7-8$ and to analzye if the transactions can be mapped to particular customer groups.

We can run a similar analysis for the offers comparing each week and each offer time slot.

{% raw %}![alt](/assets/starbucks/eda_trans_offer_periods.png){% endraw %}

### 4.3) Custom 2D bin plot

To visualize the trends in purchases, we produce a 2d data binning by time in days and by amount in set categories from 0.5 to 200$.
We visualize the bins as boxes whose color depends on the count of data points in the bin,
This is similiar to a [Spectrogram](https://en.wikipedia.org/wiki/Spectrogram), where the x axis is time, the y axis is frequency and the color is the intensity of the signal.

We add green bar to visualize the times at which the offer start.


{% raw %}![alt](/assets/starbucks/eda_trans_bin2d_outside.png){% endraw %}

This visualization shows that the purchases outside of offers are rather uniform in time and that they are concentrated in the range 1-30$.


{% raw %}![alt](/assets/starbucks/eda_trans_bin2d_offers.png){% endraw %}

The plot for the offers shows significant more purchase activity, a few hours after the offers are received.

### 4.4) Conclusions

The plot for the offers shows:
The binned plots show that the the number of purchases on average:
* increase significanlty in the offer period compared to outside the offers
* increase in for all dollar amounts but especially in the range between 2 and 10$
* increase rather quickly after an offer
* offers may have a cumulative effect: they don't seem to cancel each other 
