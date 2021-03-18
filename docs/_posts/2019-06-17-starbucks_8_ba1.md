---
  
title: "SB 6: Business Analysis:  Do offers increase revenues?"
permalink: /starbucks/starbucks_ba1_offers_vs_no_offer_periods/
excerpt: "Starbucks Case: Use of statistics to assess the profitability of promotional offers"
last_modified_at: 2019-04-18T15:53:52-04:00
toc: true
toc_label: "Page Index"
toc_sticky: true
author_profile: true
classes: post
tags:
  - Statistics
  - Visualization
categories:
  - Starbucks 
---



## Introduction

We explore the dataframe obtained in the [feature extraction section](/starbucks/starbucks_fe/) with a number of statistical and visualization tools. 
The goal is to understand if offers lead to a net increase in revenues. 


### Do offers increase the income from purchases?

We can easily compare the total hours in which an offer is viewed (net of overlaps, which are 15% of the gross offer time) with the total hours outside offers.
We find that the amounts are very simular which makes the rest of the analysis easy.

{% raw %}![alt](/assets/starbucks/eda_hours.png){% endraw %}


With the following code we can compare the income during offers (payments-rewards paid) and outside of offers.

   ```bash

	total_offers= out['Tpay_offer'].sum()
	total_rewards=out.loc[out.rewarded,'reward'].sum()

	net_income_off=total_offers-total_rewards

	total_outside= (out.groupby(by='person')['Tpay_out'].agg('mean')).sum() 
   ```

The bar shows that offers result in higher revenues, even after rewards are accounted for.

{% raw %}![alt](/assets/starbucks/eda_income.png){% endraw %}

We can go one step further and calculate for each person the net income difference between offers and no offers, i.e. the diffence between the payments during offers net of the rewards received and the payments outside of offers.
The distribution, shown below, is highly skewed but shows clearly that offers bring extra income (i.e. the mean is greater than zero). This can be determined with bootstrap: the black lines show the 99% confidence interval.
We can conclude that offers bring additional revenue of approx. 0.10-0.13 $/hour  of viewed offers.

{% raw %}![alt](/assets/starbucks/stats_avg_net_income_diff.png){% endraw %}

We can repeat the same analysis for each offer and find out the offers that are more profitable:

{% raw %}![alt](/assets/starbucks/eda_boost_offers.png){% endraw %}


### Which persons show a negative response to offers?

We define the field `neg_response` for the persons that have a negative response to offers (i.e. those that show a net revenue < $0.03).
We choose a slighly negative threshold instead of zero to avoid noise.

We find out that only 8% of the customers have a negative response to offers.
Also, comparing the income histogram for the two groups we notice that low income people tend mostly to have a positive response to offers and that people with an income between 50k and 75k may have a negative response.

{% raw %}![alt](/assets/starbucks/response_income.png){% endraw %}


Age does not seem to play a role, except maybe for those who declare an age of 118 years.

{% raw %}![alt](/assets/starbucks/response_age.png){% endraw %}

Gender seems to be equally split between those with a negative response, while Females seem to be in the majority of those with a positive response to offers

{% raw %}![alt](/assets/starbucks/response_gender.png){% endraw %}



## Conclusions

It's important to remind that due to the previous data cleaning steps only the purchases during the viewed offers concur to the revenues related to offers.
So we can conclude that: 
* offers generally lead to net incremental revenues of $0.10-0.13 per (viewed) hour of offer compared with periods without offers.
* only for 8% of the customers offers lead to smaller incremental revenues. 
* these customers who react negatively to offers seems to be more easily found between those in the 50k-75k income bracket.




