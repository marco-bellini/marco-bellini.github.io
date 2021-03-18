---
  
title: "SB 4: Exploratory Data Analysis: Offers"
permalink: /starbucks/starbucks_eda1_offers/
excerpt: "Starbucks Case: Exploring the offers, plotting basic statistics and analyzing interactions between the promotional channels"
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
In this section we focus on the offers.

We observe that the amount of offers sent to the customers are evenly distributed (No offer has been sent more times than another one).

{% raw %}![alt](/assets/starbucks/eda_offers_sent.png){% endraw %}

### Total purchases and rewards

Interestingly there is a significant spread in the amount of purchases and amount of rewards paid out across the offers.
We note that informational offers (d and f) do not pay a reward.

{% raw %}![alt](/assets/starbucks/eda_offers_purchases_rewards.png){% endraw %}

The ratio purchases/rewards shown below shows that some offers offer better returns: 

{% raw %}![alt](/assets/starbucks/eda_offers_purchases_rewards.png){% endraw %}


### Completion rate

We observe that the offers have very different completion rates (defined as the percentage of the total offers that are completed).
We consider an offer completed only if it was also viewed. If an offer was rewarded but not viewed we consider it `comp_not_viewed`.
We note that informational offers (d and f) are never marked as completed ( and do not pay a reward).

{% raw %}![alt](/assets/starbucks/eda_offers_by_offer.png){% endraw %}

#### Completed and viewed 

However it's important to understand that completed offers must be also be viewed. Below if the percentage of viewed offers: 

{% raw %}![alt](/assets/starbucks/eda_viewed_offers_by_offer.png){% endraw %}

Therefore the completed if viewed ratio is: 

{% raw %}![alt](/assets/starbucks/eda_comp_viewed_offers_by_offer.png){% endraw %}

This shows that offers have in general a quite high completion ratio but the issue is that many offers are not viewed.


#### Rewarded but not viewed 

Interestingly the data shows that offers `à` and `g` are often rewarded even if customers was not aware of the offer:
 
{% raw %}![alt](/assets/starbucks/eda_rewarded_not_viewed_offers_by_offer.png){% endraw %}


#### Effects of channel on view and completion of offers

By plotting the view rate by the the channel (only web, mobile and social, as the email channel is always used), we find that the web and social channels have a strong positive interaction. In other words the view rate increases significantly when both channels are used. 

{% raw %}![alt](/assets/starbucks/eda_offers_view_interaction.png){% endraw %}

Social and web have also a positive but smaller interaction for the completed if viewed outcome:

{% raw %}![alt](/assets/starbucks/eda_offers_completed_if_viewed_interaction.png){% endraw %}

### Conclusions

The view rates for some offers are smaller than usual: it is recommended to take advantage of the positive interaction between the web and social channel.
Offers are overlapping significantly: this makes it difficult to decouple the effects of different offers.

