---  
title: "NLP analysis of tweets during disasters"
permalink: /twitter/
excerpt: "Simple NLP techniques are used for the classification of short messages with the goal to detect disaster and distress situations."
header:
  image: /assets/nlp_disaster/twitter.jpg
  teaser: assets/nlp_disaster/twitter.jpg 
last_modified_at: 2019-04-18T15:53:52-04:00
toc: false
toc_label: "Page Index"
author_profile: true
classes: wide
---
 



The dataset is made available by [Figure Eight](https://www.figure-eight.com) and it comprises a set of messages related to disaster response, covering multiple languages, suitable for text categorization and related natural language processing tasks.

As described [here](https://www.figure-eight.com/dataset/combined-disaster-response-data/), this dataset contains 30,000 messages drawn from a number of sources covering different disasters. The data has been encoded with 36 different categories related to disaster response.

The goal it to train a ML pipeline able to detect a disaster situation from a short text message.

### [Initial Approach: Bag of Words](/twitter/initial_approach/)
Introduction, initial assessment and first approach based on a Bag of Words (B.O.W.) method.
Choice of classifier based on learning curves.
First assessment of results.

### [Refinement of Bag of Words](/twitter/bow_refinement/)
Since the classes are very imbalanced it is easy to miss relevant words in B.O.W.
An feature selection method based on undersampling of the negative class is discussed.

### [Classifier Hyperparameter Tuning](/twitter/hyperparameter/)
The hyperparameters of the classifier are tuned and conclusions are drawn

#### Tools
- scikit-learn

#### Techniques
- Bag of Words model
- feature selection
- validation curve