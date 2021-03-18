---
  
title: "SB 5: Exploratory Data Analysis: Customers"
permalink: /starbucks/starbucks_eda2_customers/
excerpt: "Starbucks Case: Exploring the customer demographics, using data binning and statistics to assess trends"
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
In this section we focus on the customers.

### Demographics

A simple histogram of the ages of the customers shows that roughly 2000 persons declared to be older than 118 years.
This could be a simple mistake (assuming the default birth data could be set to January 1st, 1900) or out of privacy concerns.
Please note that missing values are filled with `-1` to simplify visualization. 

{% raw %}![alt](/assets/starbucks/eda_customers_age.png){% endraw %}

Using a pairplot, we notice that the customers with a declared age of 118 years also declined to provide their gender and income, suggesting that they may have concerns to provide this information to the third party app.

{% raw %}![alt](/assets/starbucks/eda_customers_grid.png){% endraw %}

In the following modelling steps, it is necessary to impute missing values ("nan"), such as the missing values for gender and income, typically using the median value of the feature. 
Also, we decide not to perform data cleaning for the customers who declared an age of 118 years. We create an additional category `age_incorrect` and use the median value for the age, to avoid modelling incorrectly the impact of age.


### Total purchases during offers

We can also plot the mean amount of $ spent during an offer versus the age of the customers for each gender group.

{% raw %}![alt](/assets/starbucks/eda_customers_avg_purchase_age.png){% endraw %}


Binning the ages in intervals of 10 years, we can assert with 95% confidence that up to the age of 75 female customers spend on average more than male ones.
Unfortunately for the gender `other` the dispersion in the data and the small number of samples do not allow to draw conclusions. 

{% raw %}![alt](/assets/starbucks/eda_customers_avg_purchase_age_ci.png){% endraw %}


We can repeat the same analysis looking at the distribution by income, with similar findings.

{% raw %}![alt](/assets/starbucks/eda_customers_avg_purchase_income.png){% endraw %}

Using bins of income of 10k we can assert with 95% confidence that up to the income of 75k female customers spend on average more than male ones.


{% raw %}![alt](/assets/starbucks/eda_customers_avg_purchase_income_ci.png){% endraw %}

### Total purchases outside of offers

We can also plot the mean amount of $ spent during an offer versus the age of the customers for each gender group.

{% raw %}![alt](/assets/starbucks/eda_customers_avg_purchase_age.png){% endraw %}


Binning the ages in intervals of 10 years, we can assert with 95% confidence that up to the age of 75 female customers spend on average more than male ones.
Unfortunately for the gender `other` the dispersion in the data and the small number of samples do not allow to draw conclusions. 

{% raw %}![alt](/assets/starbucks/eda_customers_avg_purchase_age_ci.png){% endraw %}


We can repeat the same analysis looking at the distribution by income, with similar findings.

{% raw %}![alt](/assets/starbucks/eda_customers_avg_purchase_income.png){% endraw %}

Using bins of income of 10k we can assert with 95% confidence that up to the income of 75k female customers spend on average more than male ones.


{% raw %}![alt](/assets/starbucks/eda_customers_avg_purchase_income_ci.png){% endraw %}




### Completion rate

We can perform a similar analysis for the completion rate. The results are quite similar, which is not surprising considering that completion rate and mean amount of purchases during the offers are positively correlated.

{% raw %}![alt](/assets/starbucks/eda_customers_completed_income.png){% endraw %}

The data can be further analyzed evaluating confidence intervals with `proportion_confint` from statsmodels.

{% raw %}![alt](/assets/starbucks/eda_customers_completed_income_ci.png){% endraw %}


{% raw %}![alt](/assets/starbucks/eda_customers_completed_age.png){% endraw %}


{% raw %}![alt](/assets/starbucks/eda_customers_completed_age_ci.png){% endraw %}


## Conclusions

A few basic trends have been identified with statistical analysis.
It is recommended to create a feature for the people who declared an age of 118 years and to correct the corresponding value.
Also, the binned income and age could be used as a feature instead of the original variables. 



