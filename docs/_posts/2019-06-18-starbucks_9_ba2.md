---
  
title: "SB 7: Business Analysis: When an offer should not be sent?"
permalink: /starbucks/starbucks_ba2_completed_not_viewed/
excerpt: "Starbucks Case: Use of classification to decide when a promotional offer should not be sent."
last_modified_at: 2019-04-18T15:53:52-04:00
toc: true
toc_label: "Page Index"
toc_sticky: true
author_profile: true
classes: post
tags:
  - Classification
categories:
  - Starbucks 
---


## Is it better not to send offers to some customers?


One of the request from Starbucks is to identify customers who do not view offers but nonetheless complete them, receiving the reward.
Since they did not view the offer, these persons did not alter their purchase patterns but they still received a reward.


With some data aggregation we can compare the income during offers (payments-rewards paid) and outside of offers.
In particular we find out that:
* 7.4% of the total offers are rewarded even if they were not viewed
* 17% of the rewarded offers are rewarded even if they were not viewed
* in dollar terms 26257$ out of total rewards 162510$ could be saved (16.2%)
* 4.7$ on the average could be saved per offer not sent in the right situation


## How to predict which customers will be rewarded without being aware of offers

In the [Data Engineering](/starbucks/starbucks_data_engineering/) step we identified the offers that were not viewed but that were nonetheless completed and rewarded and marked them in `comp_not_viewed` column in the `combined` dataframe.
We can use the `comp_not_viewed` column as the output variable and train a classifier selecting a number of features of the `combined` dataframe.
The following derived feature used are:
* gender, income, age, user_time that are related to the customers
* offer_received (time of start of the offer), duration_hours, social, mobile, offer_type, offer_reward from the portfolio of offers
* the derived features that summarize the total $, average $ and count of payments outside of the offer time. 

No derived feature related to offers is used to avoid data leakage issues.


### Implementation and Metrics

Given the characteristics of the problems we choose the following metrics, loss functions and algorithms:
* Classes are imbalanced (7.4%) therefore it's absolutely necessary to use the option `class_weight="balanced"` to achieve sufficient precision.
* We use `RandomizedSearchCV` instead of `GridSearchCV` to explore better the parameter space.
* We initially use `scoring='roc_auc'` but it would also be possible to use a loss function based on the economic cost of False Positives and False Negatives.
* We consider both a RandomForest and a DecisionTree model because of speed of computation and interpretability of results.

### Random Forest Classifier with Average Precision metrics

We perform a search based on the following model.

   ```bash
	param_dist = {
				  "n_estimators": sp_randint(1, 40),
				  "max_depth": sp_randint(1, 20),
				  "max_features": sp_randint(1, 40),
				  "min_samples_split": sp_randint(2, 20),
				}

	# run randomized search
	n_iter_search = 30
	clf = RandomForestClassifier( bootstrap=20, criterion="entropy",class_weight="balanced")
	random_search = RandomizedSearchCV(clf,scoring='roc_auc', param_distributions=param_dist,  cv=5, iid=False,  n_iter=n_iter_search)

	random_search.fit(XT_train, y_train)
   ```


The plot below shows the ROC curve for the best RandomForest classifier with average_precision scorer.

{% raw %}![alt](/assets/starbucks/BA2_auc.png){% endraw %}

The features ranked in order of importance by the classifier are:

{% raw %}![alt](/assets/starbucks/BA2_feat_imp.png){% endraw %}


### Decision Tree Classifier with Average Precision metrics

Alternatively we can use a Decision Tree classifier, as a baseline for the more sophisticated RandomForest algorithm.

The plot below shows the ROC curve for the best DecisionTree classifier with average_precision scorer.

{% raw %}![alt](/assets/starbucks/BA2_auc_dt.png){% endraw %}

The decision graph of the best classifier is shown below:

{% raw %}![alt](/assets/starbucks/BA2_dt_graph.png){% endraw %}

As seen above, the performance of the DecisionTree is substantially identical to the RandomForest, with the added bonus that we can derive a simple and easy-to-understand classification scheme.


## Conclusions

Either the RandomForest or the DecisionTree algorithm identify pretty well the customers that should not receive an offer.

