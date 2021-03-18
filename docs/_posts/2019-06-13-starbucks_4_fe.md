---
  
title: "SB 3: Feature Extraction"
permalink: /starbucks/starbucks_fe/
excerpt: "Starbucks Case: Feature Extraction with SQL"
last_modified_at: 2019-04-18T15:53:52-04:00
toc: true
toc_label: "Page Index"
toc_sticky: true
author_profile: true
classes: post

tags:
  - Feature Engineering
categories:
  - Starbucks   
---



## Introduction

In the [previous session](/starbucks/starbucks_data_engineering/) we rearranged the data from the original data sets into more usable form. 
The next steps is to obtain a single dataframe containing both the original information and additional derived features that may be useful in solving the problem. 

The dataframes obtained from the data engineering step are combined as follows:
* the information from the `profile` database (age, gender, income,...) is matched through the person hash value.
* the information from the `portfolio` database (channel, reward, difficulty, duration,...) is matched through the offer hash value.
* the information about transactions is also added, but after aggregation and summary steps.

{% raw %}![alt](/assets/starbucks/fe_approach2.png){% endraw %}

As shown by the picture we are mainly interested in analyzing the individual offer, which is identified by a tuple: person, offer, time (i.e. offer received).  
The transactions outside of offer periods from `too` dataframe are aggregated by person. The following features are extracted:
* total $ amount of purchases outside of the offer periods (`Tpay_out`)
* number of purchase transactions outside of the offer periods (`Npay_out`)
  
The transactions during the offer periods from `tdo` dataframe are aggregated by specific offer (i.e. by matching person, offer, and starting time).
The following features are extracted:
* total $ amount of purchases within the specific offer (`Tpay_offer`)
* number of purchase transactions within the specific offer (`Npay_offer`)
* maximum and minimum $ amount of purchases within the specific offer (`Maxpay_offer`, `Maxpay_offer`)


Furthermore, the transactions during the offer periods from `tdo` dataframe are also aggregated by person.
The following features are extracted:
* total $ amount of purchases within the specific offer (`Tpay_offers_tot`)
* number of purchase transactions within the specific offer (`Tpay_offers_tot`)
* average purchases ($ and number) per unit of time (considering overlapping offers only once, as shown in the figure below)

{% raw %}![alt](/assets/starbucks/effective_duration.png){% endraw %}

The above figure shows that only viewed offers are considered and that a total offer interval is obtained through union of the viewed offers.
The union can be computed efficiently using sparse matrices and boolean operations.


##  Results
 
The final  DataFrame `out` can be seen as the all the individual promotion trials - meaning the (customer, offer, time) combinations - encountered in the promotion period. 
These combinations are associated with customer information (demographics + purchase patterns outside the offer times), offer information and the specific response to the promotion trial.
This DataFrame can be easily used for:
* statistical comparison of the effect of offers on purchasing patterns
* customer segmentation
* classification of the customer response ( if customers will view an offer or not, if they will complete it regardless of not having viewed it, ... )

 
