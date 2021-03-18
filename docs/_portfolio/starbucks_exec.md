---
  
title: "Starbucks Customer Promotion"
permalink: /starbucks/
excerpt: "Large project exploring and modelling complex behavioral data of Starbucks customers."
header:
  image: /assets/starbucks/coffee.jpg
  teaser: assets/starbucks/coffee.jpg 
last_modified_at: 2019-04-18T15:53:52-04:00
toc: false
toc_label: "Page Index"
author_profile: true
classes: wide
---



This is a very large, realistic project, provided by a collaboration of Starbucks and Udacity.
The Starbucks Capstone is a business data analysis case provided by [Udacity](https://www.udacity.com) .
It features a number of challenges in terms of data engineering and modeling.
The code for the analyis is available on [github](https://github.com/marco-bellini/Starbucks_capstone.git)

## Overview
The data provided by Starbucks simulates customer purchase behavior on a fictional App: customers buy the single product available in this data set, receive a variety of promotions, and possibly modify their buying patterns to collect the promotion reward.

## Goal

> The project itself is really to decide what next. How do we take this experimental data that's been rolling for about a month and turn that into discover which are the groups and the offers that really excite people.
The Capstone project is about discovering what is the best offer in there, not for the population as a whole but for an individual personalized level.
Certain kind of people will respond to the offer in different ways. Some people may respond negatively. 
Some people don't want to see what you send them and the best thing is not to send them something.


The goal is open. Therefore we use data exploration to understand the problem and then we identify and answer relevant business questions.

### [Instructions](/starbucks/starbucks_instructions/)
Instructions from [Udacity](https://www.udacity.com) about the capstone project.

### [Problem Statement and Approach](/starbucks/starbucks_ps/)
The problem is summarized shortly and two business-relevant questions are identified.

### [Data Engineering](/starbucks/starbucks_data_engineering/)
The original data set consists of three databases:
* promotions type and features
* customers information (gender, age, income, ...)
* transactions (purchases, reception of the offer on the App, visualization of the offer, completion of the offer and reward)

The main challenges encountered at this stage are:
* The data needs to be reshaped in a usable form. In particular time intervals are critical in this problem: it's important to know if a purchase falls within the period of validity of one offer or not, to assess whether the buying patterns are affected by the promotion.
* A superficial analysis of the data set will not show the issues of this problem. Careful analysis and custom visualization solutions are needed to explore the data set. 
 
The solutions used are:
* SQL is used to quickly separate and rearrange a number of records separated in several JSON files
* visualization is used to show time series data of the promotional offers overlap, meaning that avoiding data leakage in separating the train and test set is non-trivial

### [Feature Extraction](/starbucks/starbucks_fe/)
The previously obtained dataframes are combined and additional metrics are derived (average purchases during and outside offers,...).
SQL is used again to aggregate data. Furthermore, a large number of summary statistics are derived for further analysis and data exploration. 

### [Exploratory Data Analysis 1: Offers](/starbucks/starbucks_eda1_offers/)
Advanced statistics and visualization techniques are used to explore the offers.

### [Exploratory Data Analysis 2: Customers](/starbucks/starbucks_eda2_customers/)
Advanced statistics and visualization techniques are used to explore the customer base.

### [Exploratory Data Analysis 3: Transactions](/starbucks/starbucks_eda3_transactions/)
Advanced statistics and visualization techniques are used to explore the transaction trends.

### [Business Analysis 1](/starbucks/starbucks_ba1_offers_vs_no_offer_periods/)
In the first part of the business analysis we use statistics and bootstrap to prove that offers increase the income (net of the disbursements for rewards).
We also try to identify customers that respond negatively to offers (i.e. they purchase less).

### [Business Analysis 2](/starbucks/starbucks_ba2_completed_not_viewed/)
Classification is used to identify the customers that should not receive offers because they would complete them even without viewing them. 

### [Conclusions](/starbucks/starbucks_conclusions/)
The findings are summarized and reflections about the project are provided.


