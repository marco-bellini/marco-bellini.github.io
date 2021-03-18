---
  
title: "SB 1: Starbucks: Problem Statement"
permalink: /starbucks/starbucks_ps/
excerpt: "Starbucks Case: Overview, Problem Statement and Approach"
last_modified_at: 2019-04-18T15:53:52-04:00
toc: true
toc_label: "Page Index"
toc_sticky: true
author_profile: true
classes: post
categories:
  - Starbucks   

---

## Project Overview

The Starbucks Capstone is a business data analysis case provided by [Udacity](https://www.udacity.com) .
It features a number of challenges in terms of data engineering and modelling.

The data set is provided by Starbucks: it contains simulated customer behavior on Starbucks rewards mobile app. 
Each simulated customer receives every few days an offer (with different offer types, durations and rewards) through a variety of channels (social media, email, etc).
Not all customers receive the same offers. Customers may receive multiple offers and offers whose validity periods overlap partially. Finally, not all customers visualize an offer they receive.
The problem is simplified as in this simulated database there is only one product.

## Problem Statement

We define the problem as answering the following business questions:

1) Do offers increase revenue or now? By how much? Which offers are best in terms of bringing in incremental revenue?

2) Are some rewards wasted (i.e. are customers rewarded even if they are not aware of the offers)? How much money is wasted this way? 
Can we identify the customers that should not receive offers (those who would complete them even without viewing them)?


## Metrics

For question 1, we define the metrics as the average difference in revenues per hour between offers and period without offers.
A positive difference means that the promotions generate additional revenue (net of the rewards paid).
Bootstrap is used to evaluate the confidence interval of the difference.

For question 2, we will provide a classifier that can identify customers that complete offers even without being aware of them.
We use average precision as the metric to evaluate the performance of the classifiers, since there is significant class imbalance.
An in-depth discussion of why average precision is a good metric for cases of class imbalanced can be found on this [topic page](/topic/imbalanced_classes/)
