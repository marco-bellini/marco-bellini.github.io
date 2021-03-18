---
  
title: "SB 2: Data Engineering"
permalink: /starbucks/starbucks_data_engineering/
excerpt: "Starbucks Case: First look at the data and challenges"
last_modified_at: 2019-04-18T15:53:52-04:00
toc: true
toc_label: "Page Index"
toc_sticky: true
author_profile: true
classes: post
tags:
  - Data Engineering
  - SQL
categories:
  - Starbucks 
---




The Starbucks Capstone is a business data analysis case provided by [Udacity](https://www.udacity.com) .
It features a number of challenges in terms of data engineering and modelling.


## Introduction

The data set is provided by Starbucks: it contains simulated customer behavior on Starbucks rewards mobile app. 
Each simulated customer receives every few days an offer (with different offer types, durations and rewards) through a variety of channels (social media, email, etc).
Not all customers receive the same offers. Customers may receive multiple offers and offers whose validity periods overlap partially. Finally, not all customers visualize an offer they receive.
The problem is simplified as in this simulated database there is only one product.



## Data Engineering

This sections describes how the original data set is rearranged and combined into a more practical format.
Some preliminary exploratory data analysis is carried out to understand the data and the problem better and to extract new features that may help us in the following analysis.


The Original Data Sets comprises of three databases:

the `portfolio` database contains the 10 different offer types and their properties:


{% raw %}![alt](/assets/starbucks/sb_01_offers.png){% endraw %}


The fields have the following meaning:
*  id (string) - offer id
* offer_type (string) - type of offer ie BOGO, discount, informational
* difficulty (int) - minimum required spend to complete an offer
* reward (int) - reward given for completing an offer
* duration (int) - time for offer to be open, in days

the `profile` database contains the profiles of all the simulated users:


{% raw %}![alt](/assets/starbucks/sb_02_profile.png){% endraw %}


From the except we can notice that some users inserted incorrect information (e.g. age)


the `transcript` database contains the events logged by the app server (e.g. offers sent, transactions, rewards):
It is separated into two databases: `offers` and `transactions` 

{% raw %}![alt](/assets/starbucks/sb_03_transcript1.png){% endraw %}

Adding the data of the offer duration (converted from days to hours) from the `portfolio` database to the `offers` database, and with a bit of SQL for data extraction, we obtain a usable database with the offers received by each customer.

The time information from the columns `offer_received`,`offer_viewed`,`offer_received` of `transcript` is extracted in the `er`, `ev`, `ec` dataframes.
SQL is used to efficiently create a table that contains all the offers received by all person, with the corresponding times when offers are received, viewed and completed. 


   ```bash

	SELECT  
          er.person, er.offer, er.offer_received, ev.offer_viewed, ec.offer_completed, ec.reward, er.idx, er.offer_end
    FROM
          er 
          left join  ec on 
          er.person = ec.person and er.offer = ec.offer and ec.offer_completed <= er.offer_end and ec.offer_completed >= er.offer_received

          left join  ev on 
          er.person = ev.person and er.offer = ev.offer and ev.offer_viewed <= er.offer_end and ev.offer_viewed >= er.offer_received   and ev.offer_viewed <= ec.offer_completed
   ```

The end result is this dataframe.

{% raw %}![alt](/assets/starbucks/offer_times.png){% endraw %}

We note that in a few situations we have two offers of the same type which are both valid, and only one is viewed, as shown below. There is no way to ascertain which of the two was viewed, so we make the assumption that the most recent offer is viewed when such conflicts arise.

{% raw %}![alt](/assets/starbucks/which_offer.png){% endraw %}

Even if we have not looked at the transactions yet, this is a good time to start exploring the data.


### EDA: Offers

The data set covers one month of simulated transactions and offers. The first questions are how many offers are sent and how often.
With a simple scatterplot of `offer_received` variable the with jitter we notice that the offers are all sent out simultaneously (i.e. at hours 0, 168, 336, 408, 504, 576).
For convenience the offer_id is mapped into a,b,c... to avoid long labels in the plots.

{% raw %}![alt](/assets/starbucks/eda_00_offer_sent.png){% endraw %}

Now we represent an offer as a line with two dots (start time and end time). From this visualization we notice that most offers are in fact strongly overlapping.

{% raw %}![alt](/assets/starbucks/eda_01_offer_validity.png){% endraw %}

**Warning:** Overlapping offers means that the data needs to be treated with care in feature extraction, train-test splitting,...
{: .notice--info}

The fact that offers are overlapping has strong implications:
* if a purchase is done in the period of validity of two and more offers, to which offer should we attribute the purchase?
* how to split cleanly the data in train and test without causing data leak (it's not a straightforward issue with overlapping time series )


### EDA: Customer transactions

Now that we understand how the offers are sent, it's time to have a look to the transaction data. 
The first step is to create an appropriate visualization tool to summarize the offers and the purchases of a customer over time.

We can easily obtain the purchases from a particular customer from the `transactions` database.
The purchases made by the customer with id `78afa995795e4d85b5d9ceeca43f5fef` can be visualized as grey circles in the figure below:

{% raw %}![alt](/assets/starbucks/customer_offer_issue.png){% endraw %}

To have a full picture we also visualized the offers received by this customer.

The x axis represents the time and the top part of the graph (y>0, with white background) represents the purchases made by the customer at each time, shown as gray circles.
The lines in the lower part of the graph (y<0, with gray background) show the offers received by this customer. The transparent part of the line with the larger circles show the start and end times of the offer.
The opaque part of the line with the small dots show the period in which the offer is viewed by the customer.

Part of the goal is to determine if and how offers are increasing purchases. 
Therefore, the next task is to associate each purchase with the corresponding offer, if any.

We consider the following assumptions:
* if a purchase occurs during the effective duration of one or more offer (from viewed time to end time) the purchase is associate to all such offers
* if a purchase is made during an offer but before it was viewed or outside an offer, the purchase is not associated to any offers

And we split the `transactions` DataFrames into two:
* `tdo` : Transactions During Offers
* `too` : Transactions Outside Offers


Deriving the `tdo` database can be done efficiently with SQL code:

   ```bash

	SELECT  
            transactions.person, transactions.time, transactions.payments, offers.offer, transactions.idx  
        FROM
            transactions inner join  offers on 
            transactions.person = offers.person and transactions.time between offer_viewed and offer_end+1 
        WHERE
            viewed = 1
   ```

   

Then we can revisit the data of customer with id `78afa995795e4d85b5d9ceeca43f5fef` and highlight the offers associated with the purchases as follows:

{% raw %}![alt](/assets/starbucks/customer_offer_simple_explained.png){% endraw %}

We not the that the last two purchase fall in the period of offers i and h but only offer h is viewed at the time, so they are associated to only offer h.


Dealing with overlapping offers needs a bit more effort but the figure below shows an customer that experiences significant overlap between the offers:

{% raw %}![alt](/assets/starbucks/customer_offer_overlap1.png){% endraw %}


The `tdo` database is updated to track if a purchase is associated to several offers with the `overlaps` column ( 0 means no overlaps, 1 means two offers are overlapping, etc.) 

{% raw %}![alt](/assets/starbucks/tdo.png){% endraw %}

We also note that significant number of customers experiences overlapping offers.

{% raw %}![alt](/assets/starbucks/n_overlaps.png){% endraw %}
  


## Result

The three original data sets are not suitable for analysis but after a few transformations the new data sets are ready for [Feature Extraction](/starbucks/starbucks_fe/)
SQL has been used for the critical steps, to ensure that even a much larger database can be analyzed efficiently.  


