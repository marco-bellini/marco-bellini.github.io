---  
title: "Marketing Response and Customer Clustering"
permalink: /arvato/
excerpt: "Classification of responders to marketing campaign"
header:
  image: /assets/arvato/arvato.jpg
  teaser: assets/arvato/arvato.jpg 
last_modified_at: 2019-04-18T15:53:52-04:00
toc: false
toc_label: "Page Index"
author_profile: true
classes: wide
---
 



The dataset is provided by Arvato Financial Solutions, a Bertelsmann subsidiary and it comprises demographic attributes from the targets of a mailing order campaign. The task is to build a machine learning model that predicts whether or not each individual will respond to the campaign.

The dataset is sizeable (several hundreds of megabytes) and requires significant cleaning and feature engineering, which is omitted.
Here we focus on feature selection for categorical features and model evaluation.

### [Initial Approach](/arvato/initial_approach/)
Introduction, initial assessment, automated feature selection and model evaluation 


#### Tools
- XGBoost
- scikit-learn

#### Techniques
- contingency tables
- feature selection
- validation curve