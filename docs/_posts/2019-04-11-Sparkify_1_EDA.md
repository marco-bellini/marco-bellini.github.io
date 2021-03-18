---
title: "SP 1: EDA"
permalink: /songs/sp_eda/
excerpt: "Sparkify: EDA"
last_modified_at: 2020-08-11T15:53:52-04:00
toc: true
toc_label: "Page Index"
toc_sticky: true
author_profile: false
classes: post
tags:
  - EDA
  - Spark
categories:
  - Sparkify
---



# Sparkify Project 
This dataset comprises the user behavior data (songs played, friend requests, playlists...) of a simulated streaming music service.
The dataset (12 GB) is provided by provided by [Udacity](https://www.udacity.com) in the Data Science NanoDegree program. 
The goal is to predict the churning users.
The challenge is obviously the size of the dataset. PySpark and SQL are used to deal with the large amount of data.


# Approach

These are relavant points for the described used case
* what kind of output should the model give: classify the churned users, assign a confidence of churn
* how was the input data collected? Is it possible that the ground truth contains logging errors or spurious data (e.g. testing data, dummy accounts)?
* what performance targets should the algorithm be optimized for (high precision, high recall, ...)?
* what is the consequence of not predicting accurately churned users?

The main challenge with such a large dataset lies in efficient data analysis. 
There are several useful techniques (e.g. selecting column and rows first to reduce the amount of data, broadcasting small dataframes during joins).

Moving from the user events to summary data about users and users sessions, means that the data set is many orders of magnitude smaller and it's tractable with conventional libraries such as TensorFlow or scikit-learn.

The following steps are taken:
* data quality assessment 
* feature engineering: efficient extraction of potential useful features and summary data for users and user sessions

Next steps:
* exploratory data analysis 
* modeling
* error analysis

```python
# import libraries
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType
from pyspark.sql.types import IntegerType
from pyspark.sql.functions import desc
from pyspark.sql.functions import asc
from pyspark.sql.functions import sum as Fsum

from pyspark.sql.functions import when
import pyspark.sql.functions as F
from pyspark import StorageLevel

import datetime
import copy
import boto3
import importlib

import numpy as np
import pandas as pd
%matplotlib inline
#%load_ext sparkmonitor.kernelextension

import matplotlib.pyplot as plt

from user_agents import parse

from pyspark.sql.functions import avg, concat, count, countDistinct, \
                                col, datediff, date_format, desc, \
                                format_number, isnan, lag, lead, lit, udf, split , row_number, first, last, rank, broadcast
from pyspark.sql.functions import max as Fmax

from pyspark.sql.functions import min as Fmin

from pyspark.sql.functions import round as Fround
from pyspark.sql import Window
from pyspark.sql.types import LongType

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from datetime import datetime, timedelta
import seaborn as sns
from tabulate import tabulate

import pyspark_support as ps

# Koalas
#import databricks.koalas as ks
```

 


```python
# create a Spark session
spark = SparkSession \
    .builder \
    .appName("Spark Project") \
    .getOrCreate()

print('spark.version: ',spark.version)
print('spark.conf: \n',conf.toDebugString())
```

    spark.version:  2.4.5
    spark.conf: 
     spark.extraListeners=sparkmonitor.listener.JupyterSparkMonitorListener
    spark.driver.extraClassPath=/cluster/raid/home/chmabel1/.conda/envs/pyspark/lib/python3.7/site-packages/sparkmonitor/listener.jar


# ETL: Load and Clean Dataset <a name='LoadingData' />


There are several version of the dataset: the mini-dataset file (`mini_sparkify_event_data.json`) is a few tens megabytes. 
The full dataset `sparkify_event_data.json` is in the tens of gigabytes. It can be loaded locally of from S3.

The goal of the ETL phase is to load and clean the dataset, checking for invalid or missing data - for example, records without userids or sessionids.


```python
# json_file='mini'
# json_file='medium'
json_file='large'

```


```python
json_files={
    'mini': "mini_sparkify_event_data.json",
    'medium' : "medium-sparkify-event-data.json",
    'large' : "sparkify_event_data.json" 
}

if json_file in list(json_files.keys()):
    user_log = spark.read.json(json_files[json_file])
elif json_file=='S3':
    # Read in full sparkify dataset from Amazon S3
    event_data = "s3n://udacity-dsnd/sparkify/sparkify_event_data.json"
    user_log = spark.read.json(event_data)
    # user_log.write.parquet("s3_user_stats.parquet", mode='overwrite')
else:
    print('wrong option')
```

We use a special logging function to send the data to screen (for debugging with Jupyter notebook) or to a file local or in a cloud bucket (when running a script).


```python
# log file setup

df_logs='df'
# log_file=None
version='1.0'

# local log file
log_file=open('{}_{}_{}.log'.format(df_logs,json_file,version),'wt')

# cloud log file
# ...

```

The following function adds a few useful columns to Spark describe: number of nulls, empty strings, numerical (bigint or double), number of distinct values and list of distinct values (if they are less than 10).

The total time needed is approx 7 minutes: 2 for the describe and the rest for the column calculation ( 5s per column, 18 columns and 5 different calculations).


```python
describe_df = ps.df_summary(user_log,max_col_width=12, max_distinct=10)
```


```python
describe_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>summary</th>
      <th>count</th>
      <th>mean</th>
      <th>stddev</th>
      <th>min</th>
      <th>max</th>
      <th>dtype</th>
      <th>nulls</th>
      <th>empty_string</th>
      <th>numerical</th>
      <th>n_distinct</th>
      <th>val_distinct</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>artist</th>
      <td>20850272</td>
      <td>5.116200e+02</td>
      <td>9.686400e+02</td>
      <td>b'!!!'</td>
      <td>b'NN'</td>
      <td>string</td>
      <td>5408927</td>
      <td>0</td>
      <td>False</td>
      <td>38337</td>
      <td>too many</td>
    </tr>
    <tr>
      <th>auth</th>
      <td>26259199</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>b'Cancelled'</td>
      <td>b'Logged Out'</td>
      <td>string</td>
      <td>0</td>
      <td>0</td>
      <td>False</td>
      <td>4</td>
      <td>Logged Out,Cancelled,Guest,Logged In</td>
    </tr>
    <tr>
      <th>firstName</th>
      <td>25480720</td>
      <td>inf</td>
      <td>NaN</td>
      <td>b'Aaden'</td>
      <td>b'Zytavious'</td>
      <td>string</td>
      <td>778479</td>
      <td>0</td>
      <td>False</td>
      <td>5467</td>
      <td>too many</td>
    </tr>
    <tr>
      <th>gender</th>
      <td>25480720</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>b'F'</td>
      <td>b'M'</td>
      <td>string</td>
      <td>778479</td>
      <td>0</td>
      <td>False</td>
      <td>2</td>
      <td>F,M</td>
    </tr>
    <tr>
      <th>itemInSession</th>
      <td>26259199</td>
      <td>1.065600e+02</td>
      <td>1.176600e+02</td>
      <td>b'0'</td>
      <td>b'1428'</td>
      <td>bigint</td>
      <td>0</td>
      <td>0</td>
      <td>True</td>
      <td>1429</td>
      <td>too many</td>
    </tr>
    <tr>
      <th>lastName</th>
      <td>25480720</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>b'Abbott'</td>
      <td>b'Zuniga'</td>
      <td>string</td>
      <td>778479</td>
      <td>0</td>
      <td>False</td>
      <td>1000</td>
      <td>too many</td>
    </tr>
    <tr>
      <th>length</th>
      <td>20850272</td>
      <td>2.487300e+02</td>
      <td>9.729000e+01</td>
      <td>b'0.522'</td>
      <td>b'3024.66567'</td>
      <td>double</td>
      <td>5408927</td>
      <td>0</td>
      <td>True</td>
      <td>23748</td>
      <td>too many</td>
    </tr>
    <tr>
      <th>level</th>
      <td>26259199</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>b'free'</td>
      <td>b'paid'</td>
      <td>string</td>
      <td>0</td>
      <td>0</td>
      <td>False</td>
      <td>2</td>
      <td>free,paid</td>
    </tr>
    <tr>
      <th>location</th>
      <td>25480720</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>b'Aberdeen, SD'</td>
      <td>b'Zanesville,'</td>
      <td>string</td>
      <td>778479</td>
      <td>0</td>
      <td>False</td>
      <td>886</td>
      <td>too many</td>
    </tr>
    <tr>
      <th>method</th>
      <td>26259199</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>b'GET'</td>
      <td>b'PUT'</td>
      <td>string</td>
      <td>0</td>
      <td>0</td>
      <td>False</td>
      <td>2</td>
      <td>PUT,GET</td>
    </tr>
    <tr>
      <th>page</th>
      <td>26259199</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>b'About'</td>
      <td>b'Upgrade'</td>
      <td>string</td>
      <td>0</td>
      <td>0</td>
      <td>False</td>
      <td>22</td>
      <td>too many</td>
    </tr>
    <tr>
      <th>registration</th>
      <td>25480720</td>
      <td>1.535221e+12</td>
      <td>3.240299e+09</td>
      <td>b'150801872500'</td>
      <td>b'154382182200'</td>
      <td>bigint</td>
      <td>778479</td>
      <td>0</td>
      <td>True</td>
      <td>22247</td>
      <td>too many</td>
    </tr>
    <tr>
      <th>sessionId</th>
      <td>26259199</td>
      <td>1.005780e+05</td>
      <td>7.190921e+04</td>
      <td>b'1'</td>
      <td>b'240381'</td>
      <td>bigint</td>
      <td>0</td>
      <td>0</td>
      <td>True</td>
      <td>228713</td>
      <td>too many</td>
    </tr>
    <tr>
      <th>song</th>
      <td>20850272</td>
      <td>inf</td>
      <td>NaN</td>
      <td>b'\x18Till Kingdo'</td>
      <td>b'etta Gerist'</td>
      <td>string</td>
      <td>5408927</td>
      <td>0</td>
      <td>False</td>
      <td>253564</td>
      <td>too many</td>
    </tr>
    <tr>
      <th>status</th>
      <td>26259199</td>
      <td>2.100700e+02</td>
      <td>3.155000e+01</td>
      <td>b'200'</td>
      <td>b'404'</td>
      <td>bigint</td>
      <td>0</td>
      <td>0</td>
      <td>True</td>
      <td>3</td>
      <td>307,404,200</td>
    </tr>
    <tr>
      <th>ts</th>
      <td>26259199</td>
      <td>1.540906e+12</td>
      <td>1.515811e+09</td>
      <td>b'153835200100'</td>
      <td>b'154362240200'</td>
      <td>bigint</td>
      <td>0</td>
      <td>0</td>
      <td>True</td>
      <td>5191762</td>
      <td>too many</td>
    </tr>
    <tr>
      <th>userAgent</th>
      <td>25480720</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>b'"Mozilla/5.0'</td>
      <td>b'Mozilla/5.0'</td>
      <td>string</td>
      <td>778479</td>
      <td>0</td>
      <td>False</td>
      <td>85</td>
      <td>too many</td>
    </tr>
    <tr>
      <th>userId</th>
      <td>26259199</td>
      <td>1.488380e+06</td>
      <td>2.869701e+05</td>
      <td>b'1000025'</td>
      <td>b'1999996'</td>
      <td>string</td>
      <td>0</td>
      <td>0</td>
      <td>False</td>
      <td>22278</td>
      <td>too many</td>
    </tr>
  </tbody>
</table>
</div>





```python
# the df_describe dataframe is also saved as a text table in the log file
tab_str=tabulate(describe_df, headers='keys', tablefmt='psql', numalign="right", floatfmt="3.2e")

ps.log('ETL\nInitial Data Import\n\n',of=log_file)
ps.log(tab_str,of=log_file, skip_print=True)
ps.log('\n\n',of=log_file)

```

    ETL
    Initial Data Import
    
    
We automatically drop the rows containing null values, constructing an appropriate SQL query.
As an alternative we could have cascaded dataframe filters 


```python


cols_with_nulls=describe_df[(describe_df['nulls']>0)].index.to_list()
cond=[' {} IS NOT NULL '.format(x) for x in cols_with_nulls]
cond2='\n AND '.join(cond)
cond3="""
SELECT * FROM log
WHERE {}
""".format(cond2)
print(cond3)

```

    
    SELECT * FROM log
    WHERE  artist IS NOT NULL 
     AND  firstName IS NOT NULL 
     AND  gender IS NOT NULL 
     AND  lastName IS NOT NULL 
     AND  length IS NOT NULL 
     AND  location IS NOT NULL 
     AND  registration IS NOT NULL 
     AND  song IS NOT NULL 
     AND  userAgent IS NOT NULL 
    



```python
user_log.createOrReplaceTempView("log")
user_log = spark.sql(cond3)

describe_df = ps.df_summary(user_log,max_col_width=12, max_distinct=10)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>summary</th>
      <th>count</th>
      <th>mean</th>
      <th>stddev</th>
      <th>min</th>
      <th>max</th>
      <th>dtype</th>
      <th>nulls</th>
      <th>empty_string</th>
      <th>numerical</th>
      <th>n_distinct</th>
      <th>val_distinct</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>artist</th>
      <td>20850272</td>
      <td>5.116200e+02</td>
      <td>9.686400e+02</td>
      <td>b'!!!'</td>
      <td>b'NN'</td>
      <td>string</td>
      <td>0</td>
      <td>0</td>
      <td>False</td>
      <td>38337</td>
      <td>too many</td>
    </tr>
    <tr>
      <th>auth</th>
      <td>20850272</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>b'Logged In'</td>
      <td>b'Logged In'</td>
      <td>string</td>
      <td>0</td>
      <td>0</td>
      <td>False</td>
      <td>1</td>
      <td>Logged In</td>
    </tr>
    <tr>
      <th>firstName</th>
      <td>20850272</td>
      <td>inf</td>
      <td>NaN</td>
      <td>b'Aaden'</td>
      <td>b'Zytavious'</td>
      <td>string</td>
      <td>0</td>
      <td>0</td>
      <td>False</td>
      <td>5465</td>
      <td>too many</td>
    </tr>
    <tr>
      <th>gender</th>
      <td>20850272</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>b'F'</td>
      <td>b'M'</td>
      <td>string</td>
      <td>0</td>
      <td>0</td>
      <td>False</td>
      <td>2</td>
      <td>F,M</td>
    </tr>
    <tr>
      <th>itemInSession</th>
      <td>20850272</td>
      <td>1.088000e+02</td>
      <td>1.185900e+02</td>
      <td>b'0'</td>
      <td>b'1424'</td>
      <td>bigint</td>
      <td>0</td>
      <td>0</td>
      <td>True</td>
      <td>1423</td>
      <td>too many</td>
    </tr>
    <tr>
      <th>lastName</th>
      <td>20850272</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>b'Abbott'</td>
      <td>b'Zuniga'</td>
      <td>string</td>
      <td>0</td>
      <td>0</td>
      <td>False</td>
      <td>1000</td>
      <td>too many</td>
    </tr>
    <tr>
      <th>length</th>
      <td>20850272</td>
      <td>2.487300e+02</td>
      <td>9.729000e+01</td>
      <td>b'0.522'</td>
      <td>b'3024.66567'</td>
      <td>double</td>
      <td>0</td>
      <td>0</td>
      <td>True</td>
      <td>23748</td>
      <td>too many</td>
    </tr>
    <tr>
      <th>level</th>
      <td>20850272</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>b'free'</td>
      <td>b'paid'</td>
      <td>string</td>
      <td>0</td>
      <td>0</td>
      <td>False</td>
      <td>2</td>
      <td>free,paid</td>
    </tr>
    <tr>
      <th>location</th>
      <td>20850272</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>b'Aberdeen, SD'</td>
      <td>b'Zanesville,'</td>
      <td>string</td>
      <td>0</td>
      <td>0</td>
      <td>False</td>
      <td>886</td>
      <td>too many</td>
    </tr>
    <tr>
      <th>method</th>
      <td>20850272</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>b'PUT'</td>
      <td>b'PUT'</td>
      <td>string</td>
      <td>0</td>
      <td>0</td>
      <td>False</td>
      <td>1</td>
      <td>PUT</td>
    </tr>
    <tr>
      <th>page</th>
      <td>20850272</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>b'NextSong'</td>
      <td>b'NextSong'</td>
      <td>string</td>
      <td>0</td>
      <td>0</td>
      <td>False</td>
      <td>1</td>
      <td>NextSong</td>
    </tr>
    <tr>
      <th>registration</th>
      <td>20850272</td>
      <td>1.535220e+12</td>
      <td>3.240167e+09</td>
      <td>b'150801872500'</td>
      <td>b'154382182200'</td>
      <td>bigint</td>
      <td>0</td>
      <td>0</td>
      <td>True</td>
      <td>22231</td>
      <td>too many</td>
    </tr>
    <tr>
      <th>sessionId</th>
      <td>20850272</td>
      <td>1.017323e+05</td>
      <td>7.179624e+04</td>
      <td>b'1'</td>
      <td>b'240381'</td>
      <td>bigint</td>
      <td>0</td>
      <td>0</td>
      <td>True</td>
      <td>217969</td>
      <td>too many</td>
    </tr>
    <tr>
      <th>song</th>
      <td>20850272</td>
      <td>inf</td>
      <td>NaN</td>
      <td>b'\x18Till Kingdo'</td>
      <td>b'etta Gerist'</td>
      <td>string</td>
      <td>0</td>
      <td>0</td>
      <td>False</td>
      <td>253564</td>
      <td>too many</td>
    </tr>
    <tr>
      <th>status</th>
      <td>20850272</td>
      <td>2.000000e+02</td>
      <td>0.000000e+00</td>
      <td>b'200'</td>
      <td>b'200'</td>
      <td>bigint</td>
      <td>0</td>
      <td>0</td>
      <td>True</td>
      <td>1</td>
      <td>200</td>
    </tr>
    <tr>
      <th>ts</th>
      <td>20850272</td>
      <td>1.540919e+12</td>
      <td>1.513289e+09</td>
      <td>b'153835200100'</td>
      <td>b'154362240200'</td>
      <td>bigint</td>
      <td>0</td>
      <td>0</td>
      <td>True</td>
      <td>5105644</td>
      <td>too many</td>
    </tr>
    <tr>
      <th>userAgent</th>
      <td>20850272</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>b'"Mozilla/5.0'</td>
      <td>b'Mozilla/5.0'</td>
      <td>string</td>
      <td>0</td>
      <td>0</td>
      <td>False</td>
      <td>85</td>
      <td>too many</td>
    </tr>
    <tr>
      <th>userId</th>
      <td>20850272</td>
      <td>1.495279e+06</td>
      <td>2.885585e+05</td>
      <td>b'1000025'</td>
      <td>b'1999996'</td>
      <td>string</td>
      <td>0</td>
      <td>0</td>
      <td>False</td>
      <td>22261</td>
      <td>too many</td>
    </tr>
  </tbody>
</table>
</div>



Interestingly enough, now the `status` line contains only the 200 value (OK response), after we have removed log events with NULL values. 


```python
tab_str=tabulate(describe_df, headers='keys', tablefmt='psql', numalign="right", floatfmt="3.2e")

ps.log('After cleaning null values:\n\n',of=log_file)
ps.log(tab_str,of=log_file, skip_print=True)
ps.log('\n\n',of=log_file)
```

    After cleaning null values:



```python
describe_df.to_pickle('df_large_notnull_describe.pkl')
```



# Exploratory Data Analysis
In this section we will derive useful metrics and summary statistics about the user. Also we will aggregate events by user session to understand changes in user behavior over time.
We will define churn and look for differences in user behavior.
Since the dataset is large, we will focus on efficient operations.


## Function definition





We define a number of auxiliary functions.


These functions are used to filter data and for plotting.

```python

def convert_timestamp(df,ts_col, out_col='datetime'):
	
	df[out_col] = pd.to_datetime(df[ts_col], unit='ms')
	return df


def get_size(data):
	n_rows = data.count()
	n_columns = len(data.dtypes)
	return  n_rows,n_columns
  
def add_invalid_indicator(df,columns, postfix='invalid', custom_function=None):
	# adds an indicator to null columns
	
	for column in columns:
		if custom_function is None:
			df=df.withColumn(column+'_'+postfix, when(col(column).isNull(),True).otherwise(False) )
		elif custom_function == 'check_int':
			df=df.withColumn(column+'_'+postfix, col(column).cast("integer") )
		elif custom_function == 'is_blank_string':
			df=df.withColumn(column+'_'+postfix, (  (col(column).isNull()) | (col(column)=="" ) ) )
		else:
			df=df.withColumn(column+'_'+postfix, custom_function( col(column) ) )
	return df

def count_invalid(df, postfix='_invalid'):
	cols=df.columns
	cols_indicators=[ x for x in cols if postfix in x ]

	counts=df.select( [count( when( col(c),1)).alias(c) for c in cols_indicators ] )
	return counts

def filter_invalid(df, postfix='_invalid'):
	cols=df.columns
	cols_indicators=[ x for x in cols if postfix in x ]

	filt = ' or '.join(cols_indicators)
	filt=' not({}) '.format(filt)
	df=df.filter( filt )
	return df

def log_df_as_csv(df, log_file):
	df.toPandas().to_csv(log_file)
	return

def log(string, of=None):
	# logger
	
	if of is None:
		print(string)
	else:
		print(string)
		of.write(string)
		of.write('\n')
	return

def copy_dataframe(X):

	_schema = copy.deepcopy(X.schema)
	_schema.add('id_col', LongType(), False) # modified inplace
	_X = X.rdd.zipWithIndex().map(lambda l: list(l[0]) + [l[1]]).toDF(_schema)
	
	return _X

 
# plotting

def plot_songs_session(sessions,userId):
	"""
	plots the songs played per session by user userId
	"""
	df=sessions.filter(sessions.userId == userId).orderBy('ts_min_session', ascending=False).select(["ts_min_session","songs_session"]).toPandas()

	df=convert_timestamp(df,'ts_min_session')

	plt.figure(figsize=(12,3))
	plt.plot(df['datetime'],df['songs_session'],'o',label='user 54')
	plt.xticks(rotation=45)
	plt.ylabel('songs played per session');
	return

def get_user_level(user_log,userId):
	uf=user_log.filter(sessions.userId == userId).select(['ts','level']).toPandas()
	uf=convert_timestamp(uf,'ts')
	uf['level']=uf['level'].replace({'free':0, 'paid':1})
	return uf

def get_user_songs_per_session(user_log,userId):
	df=sessions.filter(sessions.userId == userId).orderBy('ts_min_session', ascending=False).select(["ts_min_session","songs_session"]).toPandas()
	df=convert_timestamp(df,'ts_min_session')
	return df


def plot_user_activity(user_log,userId, color='r', ax=None,alpha=0.5 ):
	uf=get_user_level(user_log,userId)
	df=get_user_songs_per_session(user_log,userId)
	
	if ax is None:
		fig=plt.figure(figsize=(10,3.5))
		ax=plt.gca()
		top_a=plt.axes([0.05,0.4,0.9,0.5])
		bottom_a=plt.axes([0.05,0.05,0.9,0.3])
	else:
		top_a, bottom_a= ax

	plt.sca(top_a)
	plt.plot(df['datetime'],df['songs_session'],'o', color=color,label='user {}'.format(userId),
			alpha=alpha)
	plt.xticks(rotation=45)
	plt.ylabel('songs played per session');
	top_a.set_xticks([],minor=[])

	plt.sca(bottom_a)
	plt.plot(uf['datetime'],uf['level'],'-',label='user {}'.format(userId), color=color,
			alpha=alpha)
	plt.xticks(rotation=45)
	plt.ylabel('free [0]/paid [1]');

	return top_a, bottom_a

```


# Data integrity tests 


```python
# filter out invalid values

user_log=add_invalid_indicator(user_log,['firstName','gender','lastName','userAgent'],custom_function = 'is_blank_string')
user_log=add_invalid_indicator(user_log,['ts','registration'])

count_na=count_invalid(user_log, postfix='_invalid')
log_df_as_csv(count_na, df_logs+'user_log_nulls')

log('file: {}'.format(json_file), of=log_file )
log('Checking invalid data (FN, G, LN, UA)', of=log_file )
log('initial rows: {}'.format(user_log.count() ), of=log_file )

user_log=filter_invalid(user_log, postfix='_invalid')
log('after filtering invalid rows: {}'.format(user_log.count() ), of=log_file )
```

    file: medium-sparkify-event-data.json
    Checking invalid data (FN, G, LN, UA)
    initial rows: 543705
    after filtering invalid rows: 528005



```python
# sample
if 0:
    fraction=0.02
    user_log=user_log.sample(False, fraction, seed=42)
```



# Feature Engineering

## Time and date, Device, States


```python
get_hour= udf( lambda x: datetime.datetime.fromtimestamp(x/1000.).hour )
get_minute= udf( lambda x: datetime.datetime.fromtimestamp(x/1000.).minute )
get_day= udf( lambda x: datetime.datetime.fromtimestamp(x/1000.).day )
get_month= udf( lambda x: datetime.datetime.fromtimestamp(x/1000.).month )
get_year= udf( lambda x: datetime.datetime.fromtimestamp(x/1000.).year )

get_device= udf( lambda x: x.split('(')[1].split(')')[0].split(';')[0] )

def extr_browser(x, with_version=True):
    if x is None:
        return('null')
    else:
        ua=parse(x)
        if with_version:
            return(ua.browser.family+'_'+ua.browser.version_string )
        else:
            return(ua.browser.family)

def extr_device_type(x):
    if x is None:
        return('null')
    else:
        us=parse(x)
        if us.is_pc:
            return('pc')
        elif us.is_mobile:
            return('mobile')
        elif us.is_bot:
            return('bot')
        elif us.is_tablet:
            return('tablet')
        else:
            return('unknown')
get_browser= udf( lambda x:    extr_browser(x, with_version=True))  
get_device_type= udf( lambda x:    extr_device_type(x))  

def extr_state(x):
    if x is None:
        return('null')
    else:
        if ',' in x:
            
            state= x.split(',')[1]
            if '-' in state:
                return state.split('-')[0]
            else:
                return state
        else:
            return x
        
get_state= udf( lambda x: extr_state(x) )

```


```python
# time
user_log= user_log.withColumn('hour',get_hour(user_log.ts) )
user_log= user_log.withColumn('day',get_day(user_log.ts) )
user_log= user_log.withColumn('month',get_month(user_log.ts) )
user_log= user_log.withColumn('year',get_year(user_log.ts) )
user_log= user_log.withColumn('minute',get_minute(user_log.ts) )

#userid
user_log= user_log.withColumn('device',get_device(user_log.userAgent) )
user_log= user_log.withColumn('browser',get_browser(user_log.userAgent) )
user_log= user_log.withColumn('device_type',get_device_type(user_log.userAgent) )

# location
user_log= user_log.withColumn('state',get_state(user_log.location) )


```


```python
flag_downgrade_event = udf( lambda x: 1 if "Submit Downgrade"  else 0, IntegerType() )
flag_cancel_event = udf( lambda x: 1 if "Cancellation Confirmation"  else 0, IntegerType() )


```

## Remove Invalid users, Top 100 Songs and Artists




```python
# user_log.select('userId').dropDuplicates().sort('userId').take(5)

```


```python
# user_log= user_log.filter( (user_log.userId !='') & (col("userId").isNotNull()) ) 
songs_v= user_log.filter( (user_log.song !='null') & (col("song").isNotNull())   ) 

# top 100 songs
ts=songs_v.groupBy('song').count().orderBy('count', ascending=False).limit(100).select("song").collect()
top100_songs = [ x['song'] for x in ts]

ts=songs_v.groupBy('artist').count().orderBy('count', ascending=False).limit(100).select("artist").collect()
top100_artist = [ x['artist'] for x in ts]


user_log=user_log.withColumn(  "top100_song",  when(user_log['song'].isin(top100_songs), 'yes').otherwise('no'))
user_log=user_log.withColumn(  "top100_artist",  when(user_log['artist'].isin(top100_artist), 'yes').otherwise('no'))

# user_log=user_log.withColumn('downgraded', flag_downgrade_event("page") )
# user_log=user_log.withColumn('churn', flag_cancel_event("page") )


# songs_v= user_log.filter(user_log.song !='null') 

log('\nUser stats:', of=log_file )
log('total users: {}'.format(user_log.count() ), of=log_file )
log('valid users: {}'.format(user_log.count() ), of=log_file )

n_unique_users=user_log.select('userId').dropDuplicates().count()
log('n_unique_users: {}'.format(n_unique_users ), of=log_file )

n_unique_sessions=user_log.select('sessionId').dropDuplicates().count()
log('n_unique_sessions: {}'.format(n_unique_sessions ), of=log_file )



```

    
    User stats:
    total users: 528005
    valid users: 528005
    n_unique_users: 448
    n_unique_sessions: 4470


## User Sessions  <a name='UserSessions' />


```python
cnt_cond = lambda cond: F.sum(F.when(cond, 1).otherwise(0))
```


```python
sessions= user_log.groupBy('userId','SessionId').agg( Fmin('ts').alias('ts_min_session'),Fmax('ts').alias('ts_max_session'), count('song').alias('songs_session'), Fsum('length').alias('sum_length_songs'),
                                                  cnt_cond(col("page") == "Roll Advert" ).alias('n_ads'), cnt_cond(col("page") == "Logout" ).alias('n_logouts'), 
                                                  cnt_cond(col("page") == "Add Friend" ).alias('n_add_friend'),cnt_cond(col("page") == "Add to Playlist" ).alias('n_add_playlist') ,
                                                  cnt_cond(col("page") == "Thumbs Up" ).alias('t_up'), cnt_cond(col("page") == "Thumbs Down" ).alias('t_down'),
                                                  cnt_cond(col("page") == "Submit Downgrade" ).alias('down'),cnt_cond(col("page") == "Submit Upgrade" ).alias('up'),
                                                  cnt_cond(col("page") == "Cancel" ).alias('cancel'),cnt_cond(col("page") == "Cancellation Confirmation" ).alias('confirm cancel') )
```


```python
sessions=sessions.withColumn( "session_length", (col("ts_max_session") - col("ts_min_session") )/(1000.*60 )  )

# looks strange.
# sessions=sessions.withColumn( "perc_compl_songs", 100.*((col("ts_max_session") - col("ts_min_session") )/(1000. )  )/ col("sum_length_songs")  )

sessions=sessions.withColumn( "ads/session min", (col("n_ads") / col("session_length")  ) )
sessions=sessions.withColumn( "t_up/session min", (col("t_up") / col("session_length")  ) )
sessions=sessions.withColumn( "t_down/session min", (col("t_down") / col("session_length")  ) )
sessions=sessions.withColumn( "friends/session min", (col("n_add_friend") / col("session_length")  ) )
sessions=sessions.withColumn( "playlist/session min", (col("n_add_playlist") / col("session_length")  ) )
sessions=sessions.withColumn( "logouts/session min", (col("n_logouts") / col("session_length")  ))
                             
                             
sessions=sessions.fillna(0)
sessions.persist()                             

log('\nUser sessions:', of=log_file )
log('total sessions: {}'.format(sessions.count() ), of=log_file )
```

    
    User sessions:
    total sessions: 6080



```python
# average sessions over users

user_sessions=sessions.groupBy('userId').agg(avg('ads/session min').alias('avg_ads/session min'),avg('t_up/session min').alias('avg_t_up/session min'),
                                             avg('t_down/session min').alias('avg_t_down/session min'),
                                             avg('friends/session min').alias('avg_friends/session min'),avg('playlist/session min').alias('avg_playlist/session min'),
                                             avg('logouts/session min').alias('avg_logouts/session min') )
```


```python
userId= 54
plot_songs_session(sessions,userId)
```


{% raw %}![alt](/assets/UCI_Credit_Cards_fairness/output_46_0.png){% endraw %}



```python

userId=54
color='r'
top_a, bottom_a=plot_user_activity(user_log,userId, color=color)
userId=53
color='b'
plot_user_activity(user_log,userId, color=color, ax=[top_a, bottom_a]);
top_a.legend(loc=2);          
              
# TODO: plot cancellation event?    
```


{% raw %}![alt](/assets/UCI_Credit_Cards_fairness/output_47_0.png){% endraw %}



```python
user_log.select('page').dropDuplicates().sort("page").show()
```

    +--------------------+
    |                page|
    +--------------------+
    |               About|
    |          Add Friend|
    |     Add to Playlist|
    |              Cancel|
    |Cancellation Conf...|
    |           Downgrade|
    |               Error|
    |                Help|
    |                Home|
    |              Logout|
    |            NextSong|
    |         Roll Advert|
    |       Save Settings|
    |            Settings|
    |    Submit Downgrade|
    |      Submit Upgrade|
    |         Thumbs Down|
    |           Thumbs Up|
    |             Upgrade|
    +--------------------+
    


## User statistics  <a name='UserStatistics' />


```python
# select(['userId','song','ts','page','artist','length' ]) speeds up 25%
user_stats = user_log.select(['userId','song','ts','page','artist','length' ]).groupBy('userId').agg( countDistinct ('song').alias('different_songs_per_user'),
                count('song').alias('total_songs_per_user') ,countDistinct ('artist').alias('different_artists_per_user'),
                cnt_cond(col("page") == "Cancellation Confirmation" ).alias('churn'),
                                            
                Fmin('ts').alias('ts_min'),Fmax('ts').alias('ts_max'), Fsum('length').alias('sum_length_songs'),
                  cnt_cond(col("page") == "Roll Advert" ).alias('n_ads'), cnt_cond(col("page") == "Logout" ).alias('n_logouts'), 
                  cnt_cond(col("page") == "Add Friend" ).alias('n_add_friend'),cnt_cond(col("page") == "Add to Playlist" ).alias('n_add_playlist') ,
                  cnt_cond(col("page") == "Thumbs Up" ).alias('t_up'), cnt_cond(col("page") == "Thumbs Down" ).alias('t_down'),
                  cnt_cond(col("page") == "Submit Downgrade" ).alias('down'),cnt_cond(col("page") == "Submit Upgrade" ).alias('up'),
                  cnt_cond(col("page") == "Cancel" ).alias('cancel'),cnt_cond(col("page") == "Cancellation Confirmation" ).alias('confirm_cancel')                                            
                                           )   

```


```python
# save the recomputation
# user_stats_copy =copy_dataframe(user_stats)
# user_stats_copy.take(3)
```


```python
# reload:
# user_stats =copy_dataframe(user_stats_copy)
```


```python

user_stats=user_stats.withColumn( "total_time", (col("ts_max") - col("ts_min")  ) )


# in hours
user_stats=user_stats.withColumn( "total_time_h", (col("ts_max") - col("ts_min")  )/(1000.*60.*60. ) )

user_stats=user_stats.withColumn( "ads/min", (col("n_ads") / col("total_time_h")  ) )
user_stats=user_stats.withColumn( "t_up/min", (col("t_up") / col("total_time_h")  ) )
user_stats=user_stats.withColumn( "t_down/min", (col("t_down") / col("total_time_h")  ) )
user_stats=user_stats.withColumn( "friends/min", (col("n_add_friend") / col("total_time_h")  ) )
user_stats=user_stats.withColumn( "playlist/min", (col("n_add_playlist") / col("total_time_h")  ) )
user_stats=user_stats.withColumn( "logouts/min", (col("n_logouts") / col("total_time_h")  ))
                             

log('Generate user statistics', of=log_file )
log('1 GroupBy: n_unique_users: {}'.format(user_stats.select('userId').dropDuplicates().count() ), of=log_file )

most_repetitions_per_song=(songs_v.groupBy('userId','song').count().groupBy('userId').max())
most_repetitions_per_song=most_repetitions_per_song.withColumnRenamed('max(count)','most repetitions per song')



```

    Generate user statistics
    1 GroupBy: n_unique_users: 448



```python
# broadcasting small DF speeds up the process

user_stats=user_stats.join(broadcast(most_repetitions_per_song), on='userId', how='left')
log('2 Join M rep/song: n_unique_users: {}'.format(n_unique_users ), of=log_file )

user_stats=user_stats.join( broadcast(user_sessions) , on='userId', how='left')
user_stats.collect()
log('3 Join user_sessions: n_unique_users: {}'.format(user_stats.select('userId').dropDuplicates().count() ), of=log_file )
```

    2 Join M rep/song: n_unique_users: 448
    3 Join user_sessions: n_unique_users: 448



```python
# user_stats=user_stats.join((most_repetitions_per_song), on='userId', how='left')
# log('2 Join M rep/song: n_unique_users: {}'.format(n_unique_users ), of=log_file )

# user_stats=user_stats.join((user_sessions) , on='userId', how='left')
# user_stats.collect()

# log('3 Join user_sessions: n_unique_users: {}'.format(user_stats.select('userId').dropDuplicates().count() ), of=log_file )
```


```python
#user_min_ts = user_log.select('userId','ts').groupBy('userId').min('ts').withColumnRenamed('min(ts)','join_time')
user_downgrade_ts = user_log.filter(user_log.page== "Submit Downgrade").select('userId','ts','page').groupBy('userId').max('ts').withColumnRenamed('max(ts)','last_downgrade_time')
user_upgrade_ts = user_log.filter(user_log.page== "Submit Upgrade").select('userId','ts','page').groupBy('userId').min('ts').withColumnRenamed('min(ts)','first_upgrade_time')

max_time = user_log.agg({"ts": "max"}).collect()[0][0]
min_time = user_log.agg({"ts": "min"}).collect()[0][0]

# user_stats=user_stats.join(user_min_ts, on='userId', how='left')
# log('4 Join min ts: n_unique_users: {}'.format(user_stats.select('userId').dropDuplicates().count() ), of=log_file )
user_stats=user_stats.join(broadcast(user_downgrade_ts), on='userId', how='left')
log('5 Join down ts: n_unique_users: {}'.format(user_stats.select('userId').dropDuplicates().count() ), of=log_file )
user_stats=user_stats.join(broadcast(user_upgrade_ts), on='userId', how='left')
log('5 Join up ts: u n_unique_users: {}'.format(user_stats.select('userId').dropDuplicates().count() ), of=log_file )

user_stats=user_stats.withColumn(  "time_before_upgrade",  user_stats['last_downgrade_time'] - min_time)
log('6 Join time before up: n_unique_users: {}'.format(user_stats.select('userId').dropDuplicates().count() ), of=log_file )
# this does not work so well
# user_stats=user_stats.withColumn(  "time_premium",  user_stats['last_downgrade_time'] - user_stats['first_upgrade_time'] )

# user_stats=user_stats.withColumn(  "total_time",  max_time - user_stats['join_time'] )
# log('4 Join total time: n_unique_users: {}'.format(user_stats.select('userId').dropDuplicates().count() ), of=log_file )

# obsolete
## time_before_upgrade = user_upgrade_ts.select('last_downgrade_time').subtract(min_time) 
## time_premium = user_downgrade_ts - user_upgrade_ts   



```

    5 Join down ts: n_unique_users: 448
    5 Join up ts: u n_unique_users: 448
    6 Join time before up: n_unique_users: 448


### User service level at beginning and at the end


```python
# user_stats_copy =copy_dataframe(user_stats)
# reload:
# user_stats =copy_dataframe(user_stats_copy)
```


```python
# windowing is very fast
# beginning level
w_ts_asc = Window.partitionBy("userId" ).orderBy(asc("ts")  ) 
user_beg_level=user_log.select('*', row_number().over(w_ts_asc).alias('row')).where(col('row') == 1)


# final level
w_ts_desc = Window.partitionBy("userId" ).orderBy(desc("ts")  ) 
user_end_level=user_log.select('*', row_number().over(w_ts_desc).alias('row')).where(col('row') == 1)

user_beg_level=user_beg_level.withColumnRenamed('level','beg_level')
user_end_level=user_end_level.withColumnRenamed('level','end_level')

# combine select and broadcast
user_stats=user_stats.join(broadcast(user_beg_level.select(['userId','beg_level'])), on='userId', how='left')
log('7 Join beg_level: n_unique_users: {}'.format(user_stats.select('userId').dropDuplicates().count() ), of=log_file )
user_stats=user_stats.join(broadcast(user_end_level.select(['userId','end_level'])), on='userId', how='left')
log('8 Join end_level: n_unique_users: {}'.format(user_stats.select('userId').dropDuplicates().count() ), of=log_file )

```

    7 Join beg_level: n_unique_users: 448
    8 Join end_level: n_unique_users: 448



```python
# visualization
# user_beg_level.filter(col('userId').isin([53,54])).select(['userId','ts','beg_level','row']).orderBy(['userId','ts']).show() 
# user_end_level.filter( col('userId').isin([53,54])).select(['userId','ts','end_level','row']).orderBy(['userId','ts']).show() 

```

### Amount of paid and free service:  <a name='PaidFreeTime' />

speedup with select and broadcast


```python
w_ts_asc = Window.partitionBy("userId" ).orderBy(asc("ts")  )

user_transitions= user_log.filter(  (col("page")=="Submit Upgrade") | (col("page")=="Submit Downgrade")).withColumn("prev_ts", lag('ts', 1).over(w_ts_asc)).fillna(0)
user_transitions=user_transitions.withColumn('interval', col('ts')-col('prev_ts'))
user_transitions=user_transitions.join( broadcast(user_stats.select(
    ["userId","ts_min","last_downgrade_time","first_upgrade_time","time_before_upgrade","beg_level","end_level"]) ) , on='userId') 

user_transitions=user_transitions.withColumn('level_num', when( col('level')=='paid',1).otherwise(0)  ) 
user_transitions=user_transitions.withColumn('paid_time',   user_transitions['level_num']*user_transitions['interval'] ) 
user_transitions=user_transitions.fillna(0,subset=['paid_time'])

# this still lacks the last time period
user_transitions_g=user_transitions.groupBy('userId').agg( Fsum('paid_time').alias('paid_time') ,Fmax('ts').alias('last_ts_change'), last('level_num').alias('last_level_num') )
user_transitions_g=user_transitions_g.fillna(0,subset=['paid_time'])

# it's added now
user_transitions_g=user_transitions_g.withColumn('paid_time',user_transitions_g['paid_time']+user_transitions_g['last_ts_change']*user_transitions_g['last_level_num'])

user_stats=user_stats.join( broadcast(user_transitions_g.select('userId','paid_time')),on='userId', how='left')
user_stats=user_stats.withColumn('frac_time_paid',100.*user_stats['paid_time']/user_stats['total_time'] )

log('9 Join transitions: n_unique_users: {}'.format(user_stats.select('userId').dropDuplicates().count() ), of=log_file )
```

    9 Join transitions: n_unique_users: 448



```python
multiple_upgrades=  user_log.filter(user_log.page== "Submit Upgrade").select('userId','page').groupby('userId').count()
multiple_downgrades=  user_log.filter(user_log.page== "Submit Downgrade").select('userId','page').groupby('userId').count()

log('users with 2 or more upgrades   : {}'.format(multiple_upgrades.filter( "count >1").count() ), of=log_file )
log('users with 2 or more downgrades : {}'.format(multiple_downgrades.filter( "count >1").count() ), of=log_file )

```

    users with 2 or more upgrades   : 47
    users with 2 or more downgrades : 19



```python
# user_log.printSchema()
```


```python

songs_v2= user_log.filter(user_log.song !='null').select('userId','song','top100_song','top100_artist').dropDuplicates()

n_top100_songs=songs_v2.filter(songs_v2.top100_song=='yes').groupBy('userId').count().withColumnRenamed('count','n_top100_songs') 
n_top100_artists=songs_v2.filter(songs_v2.top100_artist=='yes').groupBy('userId').count().withColumnRenamed('count','n_top100_artists')

user_stats=user_stats.join( broadcast(n_top100_songs) , on='userId', how='left')
log('10 Join top100 songs: n_unique_users: {}'.format(user_stats.select('userId').dropDuplicates().count() ), of=log_file )

user_stats=user_stats.join( broadcast(n_top100_artists) , on='userId', how='left')
log('11 Join top100 artists: n_unique_users: {}'.format(user_stats.select('userId').dropDuplicates().count() ), of=log_file )

user_stats=user_stats.withColumn('perc_songs_top100',100.*user_stats['n_top100_songs']/user_stats['different_songs_per_user'] )
user_stats=user_stats.withColumn('perc_artists_top100',100.*user_stats['n_top100_artists']/user_stats['different_artists_per_user'] )

```

    10 Join top100 songs: n_unique_users: 448
    11 Join top100 artists: n_unique_users: 448



```python
   
avg_len=songs_v.groupBy('userId').agg(F.avg("length").alias('avg_length'))
std_len=songs_v.groupBy('userId').agg({"length":'stddev'}).alias('std_length')

user_stats=user_stats.join( broadcast(avg_len), on='userId', how='left')
log('11 Join avg song length: n_unique_users: {}'.format(user_stats.select('userId').dropDuplicates().count() ), of=log_file )
user_stats=user_stats.join( broadcast(std_len), on='userId', how='left')
log('12 Join std song length: n_unique_users: {}'.format(user_stats.select('userId').dropDuplicates().count() ), of=log_file )

user_stats=user_stats.fillna(0)
```

    11 Join avg song length: n_unique_users: 448
    12 Join std song length: n_unique_users: 448



```python
# user_stats=user_stats.fillna(0)
user_stats.printSchema()
```

    root
     |-- userId: string (nullable = true)
     |-- different_songs_per_user: long (nullable = false)
     |-- total_songs_per_user: long (nullable = false)
     |-- different_artists_per_user: long (nullable = false)
     |-- churn: long (nullable = true)
     |-- ts_min: long (nullable = true)
     |-- ts_max: long (nullable = true)
     |-- sum_length_songs: double (nullable = false)
     |-- n_ads: long (nullable = true)
     |-- n_logouts: long (nullable = true)
     |-- n_add_friend: long (nullable = true)
     |-- n_add_playlist: long (nullable = true)
     |-- t_up: long (nullable = true)
     |-- t_down: long (nullable = true)
     |-- down: long (nullable = true)
     |-- up: long (nullable = true)
     |-- cancel: long (nullable = true)
     |-- confirm cancel: long (nullable = true)
     |-- total_time: long (nullable = true)
     |-- total_time_h: double (nullable = false)
     |-- ads/min: double (nullable = false)
     |-- t_up/min: double (nullable = false)
     |-- t_down/min: double (nullable = false)
     |-- friends/min: double (nullable = false)
     |-- playlist/min: double (nullable = false)
     |-- logouts/min: double (nullable = false)
     |-- most repetitions per song: long (nullable = true)
     |-- avg_ads/session min: double (nullable = false)
     |-- avg_t_up/session min: double (nullable = false)
     |-- avg_t_down/session min: double (nullable = false)
     |-- avg_friends/session min: double (nullable = false)
     |-- avg_playlist/session min: double (nullable = false)
     |-- avg_logouts/session min: double (nullable = false)
     |-- last_downgrade_time: long (nullable = true)
     |-- first_upgrade_time: long (nullable = true)
     |-- time_before_upgrade: long (nullable = true)
     |-- beg_level: string (nullable = true)
     |-- end_level: string (nullable = true)
     |-- paid_time: long (nullable = true)
     |-- frac_time_paid: double (nullable = false)
     |-- n_top100_songs: long (nullable = true)
     |-- n_top100_artists: long (nullable = true)
     |-- perc_songs_top100: double (nullable = false)
     |-- perc_artists_top100: double (nullable = false)
     |-- avg_length: double (nullable = false)
     |-- stddev(length): double (nullable = false)
    



```python
# user_stats.persist(StorageLevel.MEMORY_AND_DISK)
log('\nfinished user_stats: n_rows {}'.format( user_stats.count() ), of=log_file )
log('n_unique_users: {}'.format(user_stats.select('userId').dropDuplicates().count() ), of=log_file )
```

    
    finished user_stats: n_rows 448
    n_unique_users: 448



```python
user_stats.select(['total_time','total_time_h','paid_time','frac_time_paid']).take(5)
```




    [Row(total_time=262926000, total_time_h=73.035, paid_time=0, frac_time_paid=0.0),
     Row(total_time=2483596000, total_time_h=689.8877777777777, paid_time=0, frac_time_paid=0.0),
     Row(total_time=709483000, total_time_h=197.07861111111112, paid_time=0, frac_time_paid=0.0),
     Row(total_time=3776668000, total_time_h=1049.0744444444445, paid_time=0, frac_time_paid=0.0),
     Row(total_time=3541953000, total_time_h=983.8758333333334, paid_time=0, frac_time_paid=0.0)]






# EDA  

## Count null and zeros


```python
def nan_percentage(df, normalize=True):
    res=df.select([count(when(isnan(c), c)).alias(c) for c in df.columns]).toPandas()
    n_lines= df.count()
    if normalize:
        res=100.*res/n_lines
        res=res.rename(index={0:'perc. nan'})
    else:
        res=res.rename(index={0:'n. nan'})
    return res

def zero_percentage(df, normalize=True):
    res=df.select([count(when(col(c)==0., c)).alias(c) for c in df.columns]).toPandas()
    n_lines= df.count()
    if normalize:
        res=100.*res/n_lines
        res=res.rename(index={0:'perc. zeros'})
    else:
        res=res.rename(index={0:'n. zeros'})
    return res

```


```python
res=zero_percentage(user_stats)
```


```python
res.T
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>perc. zeros</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>userId</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>different_songs_per_user</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>total_songs_per_user</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>different_artists_per_user</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>churn</th>
      <td>77.901786</td>
    </tr>
    <tr>
      <th>ts_min</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>ts_max</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>sum_length_songs</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>n_ads</th>
      <td>11.830357</td>
    </tr>
    <tr>
      <th>n_logouts</th>
      <td>4.910714</td>
    </tr>
    <tr>
      <th>n_add_friend</th>
      <td>8.705357</td>
    </tr>
    <tr>
      <th>n_add_playlist</th>
      <td>4.464286</td>
    </tr>
    <tr>
      <th>t_up</th>
      <td>2.008929</td>
    </tr>
    <tr>
      <th>t_down</th>
      <td>13.169643</td>
    </tr>
    <tr>
      <th>down</th>
      <td>78.348214</td>
    </tr>
    <tr>
      <th>up</th>
      <td>47.991071</td>
    </tr>
    <tr>
      <th>cancel</th>
      <td>77.901786</td>
    </tr>
    <tr>
      <th>confirm cancel</th>
      <td>77.901786</td>
    </tr>
    <tr>
      <th>total_time</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>total_time_h</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>ads/min</th>
      <td>11.830357</td>
    </tr>
    <tr>
      <th>t_up/min</th>
      <td>2.008929</td>
    </tr>
    <tr>
      <th>t_down/min</th>
      <td>13.169643</td>
    </tr>
    <tr>
      <th>friends/min</th>
      <td>8.705357</td>
    </tr>
    <tr>
      <th>playlist/min</th>
      <td>4.464286</td>
    </tr>
    <tr>
      <th>logouts/min</th>
      <td>4.910714</td>
    </tr>
    <tr>
      <th>most repetitions per song</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>avg_ads/session min</th>
      <td>11.830357</td>
    </tr>
    <tr>
      <th>avg_t_up/session min</th>
      <td>2.008929</td>
    </tr>
    <tr>
      <th>avg_t_down/session min</th>
      <td>13.169643</td>
    </tr>
    <tr>
      <th>avg_friends/session min</th>
      <td>8.705357</td>
    </tr>
    <tr>
      <th>avg_playlist/session min</th>
      <td>4.464286</td>
    </tr>
    <tr>
      <th>avg_logouts/session min</th>
      <td>4.910714</td>
    </tr>
    <tr>
      <th>last_downgrade_time</th>
      <td>78.348214</td>
    </tr>
    <tr>
      <th>first_upgrade_time</th>
      <td>47.991071</td>
    </tr>
    <tr>
      <th>time_before_upgrade</th>
      <td>78.348214</td>
    </tr>
    <tr>
      <th>beg_level</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>end_level</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>paid_time</th>
      <td>78.348214</td>
    </tr>
    <tr>
      <th>frac_time_paid</th>
      <td>78.348214</td>
    </tr>
    <tr>
      <th>n_top100_songs</th>
      <td>1.116071</td>
    </tr>
    <tr>
      <th>n_top100_artists</th>
      <td>0.223214</td>
    </tr>
    <tr>
      <th>perc_songs_top100</th>
      <td>1.116071</td>
    </tr>
    <tr>
      <th>perc_artists_top100</th>
      <td>0.223214</td>
    </tr>
    <tr>
      <th>avg_length</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>stddev(length)</th>
      <td>0.223214</td>
    </tr>
  </tbody>
</table>
</div>




```python
uf= user_stats.toPandas()


```


```python
uf.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 448 entries, 0 to 447
    Data columns (total 46 columns):
     #   Column                      Non-Null Count  Dtype  
    ---  ------                      --------------  -----  
     0   userId                      448 non-null    object 
     1   different_songs_per_user    448 non-null    int64  
     2   total_songs_per_user        448 non-null    int64  
     3   different_artists_per_user  448 non-null    int64  
     4   churn                       448 non-null    int64  
     5   ts_min                      448 non-null    int64  
     6   ts_max                      448 non-null    int64  
     7   sum_length_songs            448 non-null    float64
     8   n_ads                       448 non-null    int64  
     9   n_logouts                   448 non-null    int64  
     10  n_add_friend                448 non-null    int64  
     11  n_add_playlist              448 non-null    int64  
     12  t_up                        448 non-null    int64  
     13  t_down                      448 non-null    int64  
     14  down                        448 non-null    int64  
     15  up                          448 non-null    int64  
     16  cancel                      448 non-null    int64  
     17  confirm cancel              448 non-null    int64  
     18  total_time                  448 non-null    int64  
     19  total_time_h                448 non-null    float64
     20  ads/min                     448 non-null    float64
     21  t_up/min                    448 non-null    float64
     22  t_down/min                  448 non-null    float64
     23  friends/min                 448 non-null    float64
     24  playlist/min                448 non-null    float64
     25  logouts/min                 448 non-null    float64
     26  most repetitions per song   448 non-null    int64  
     27  avg_ads/session min         448 non-null    float64
     28  avg_t_up/session min        448 non-null    float64
     29  avg_t_down/session min      448 non-null    float64
     30  avg_friends/session min     448 non-null    float64
     31  avg_playlist/session min    448 non-null    float64
     32  avg_logouts/session min     448 non-null    float64
     33  last_downgrade_time         448 non-null    int64  
     34  first_upgrade_time          448 non-null    int64  
     35  time_before_upgrade         448 non-null    int64  
     36  beg_level                   448 non-null    object 
     37  end_level                   448 non-null    object 
     38  paid_time                   448 non-null    int64  
     39  frac_time_paid              448 non-null    float64
     40  n_top100_songs              448 non-null    int64  
     41  n_top100_artists            448 non-null    int64  
     42  perc_songs_top100           448 non-null    float64
     43  perc_artists_top100         448 non-null    float64
     44  avg_length                  448 non-null    float64
     45  stddev(length)              448 non-null    float64
    dtypes: float64(19), int64(24), object(3)
    memory usage: 161.1+ KB



```python
# uf.describe().T

uf.to_pickle('medium_user_stats.pkl')
```


```python
uf.groupby('churn').mean().T
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>churn</th>
      <th>0</th>
      <th>1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>different_songs_per_user</th>
      <td>8.793782e+02</td>
      <td>7.566162e+02</td>
    </tr>
    <tr>
      <th>total_songs_per_user</th>
      <td>9.986189e+02</td>
      <td>8.521111e+02</td>
    </tr>
    <tr>
      <th>different_artists_per_user</th>
      <td>6.777708e+02</td>
      <td>5.927374e+02</td>
    </tr>
    <tr>
      <th>ts_min</th>
      <td>1.538977e+12</td>
      <td>1.538660e+12</td>
    </tr>
    <tr>
      <th>ts_max</th>
      <td>1.542957e+12</td>
      <td>1.540849e+12</td>
    </tr>
    <tr>
      <th>sum_length_songs</th>
      <td>2.482415e+05</td>
      <td>2.121705e+05</td>
    </tr>
    <tr>
      <th>n_ads</th>
      <td>1.600860e+01</td>
      <td>2.208081e+01</td>
    </tr>
    <tr>
      <th>n_logouts</th>
      <td>1.377077e+01</td>
      <td>1.195960e+01</td>
    </tr>
    <tr>
      <th>n_add_friend</th>
      <td>1.876791e+01</td>
      <td>1.552525e+01</td>
    </tr>
    <tr>
      <th>n_add_playlist</th>
      <td>2.891117e+01</td>
      <td>2.281818e+01</td>
    </tr>
    <tr>
      <th>t_up</th>
      <td>5.594842e+01</td>
      <td>4.343434e+01</td>
    </tr>
    <tr>
      <th>t_down</th>
      <td>1.075931e+01</td>
      <td>1.167677e+01</td>
    </tr>
    <tr>
      <th>down</th>
      <td>2.521490e-01</td>
      <td>2.929293e-01</td>
    </tr>
    <tr>
      <th>up</th>
      <td>6.217765e-01</td>
      <td>7.070707e-01</td>
    </tr>
    <tr>
      <th>cancel</th>
      <td>0.000000e+00</td>
      <td>1.000000e+00</td>
    </tr>
    <tr>
      <th>confirm cancel</th>
      <td>0.000000e+00</td>
      <td>1.000000e+00</td>
    </tr>
    <tr>
      <th>total_time</th>
      <td>3.980018e+09</td>
      <td>2.189237e+09</td>
    </tr>
    <tr>
      <th>total_time_h</th>
      <td>1.105561e+03</td>
      <td>6.081214e+02</td>
    </tr>
    <tr>
      <th>ads/min</th>
      <td>4.506754e-02</td>
      <td>1.816119e-01</td>
    </tr>
    <tr>
      <th>t_up/min</th>
      <td>6.931971e-02</td>
      <td>1.332509e-01</td>
    </tr>
    <tr>
      <th>t_down/min</th>
      <td>3.307071e-02</td>
      <td>3.364564e-02</td>
    </tr>
    <tr>
      <th>friends/min</th>
      <td>2.648617e-02</td>
      <td>4.630488e-02</td>
    </tr>
    <tr>
      <th>playlist/min</th>
      <td>3.601300e-02</td>
      <td>5.851843e-02</td>
    </tr>
    <tr>
      <th>logouts/min</th>
      <td>4.597222e-02</td>
      <td>4.114226e-02</td>
    </tr>
    <tr>
      <th>most repetitions per song</th>
      <td>6.564470e+00</td>
      <td>5.919192e+00</td>
    </tr>
    <tr>
      <th>avg_ads/session min</th>
      <td>2.135522e-02</td>
      <td>1.804753e-02</td>
    </tr>
    <tr>
      <th>avg_t_up/session min</th>
      <td>4.282384e-02</td>
      <td>1.299443e-02</td>
    </tr>
    <tr>
      <th>avg_t_down/session min</th>
      <td>6.771189e-02</td>
      <td>5.024829e-02</td>
    </tr>
    <tr>
      <th>avg_friends/session min</th>
      <td>5.951062e-02</td>
      <td>5.573604e-03</td>
    </tr>
    <tr>
      <th>avg_playlist/session min</th>
      <td>1.893719e-02</td>
      <td>6.043352e-03</td>
    </tr>
    <tr>
      <th>avg_logouts/session min</th>
      <td>2.029265e-01</td>
      <td>1.737031e-01</td>
    </tr>
    <tr>
      <th>last_downgrade_time</th>
      <td>3.356450e+11</td>
      <td>3.267643e+11</td>
    </tr>
    <tr>
      <th>first_upgrade_time</th>
      <td>7.855859e+11</td>
      <td>8.553669e+11</td>
    </tr>
    <tr>
      <th>time_before_upgrade</th>
      <td>6.457149e+08</td>
      <td>4.472094e+08</td>
    </tr>
    <tr>
      <th>paid_time</th>
      <td>2.476477e+11</td>
      <td>2.646431e+11</td>
    </tr>
    <tr>
      <th>frac_time_paid</th>
      <td>6.410535e+03</td>
      <td>1.167257e+04</td>
    </tr>
    <tr>
      <th>n_top100_songs</th>
      <td>4.414613e+01</td>
      <td>4.014141e+01</td>
    </tr>
    <tr>
      <th>n_top100_artists</th>
      <td>1.599284e+02</td>
      <td>1.397273e+02</td>
    </tr>
    <tr>
      <th>perc_songs_top100</th>
      <td>7.038503e+00</td>
      <td>7.410120e+00</td>
    </tr>
    <tr>
      <th>perc_artists_top100</th>
      <td>2.317508e+01</td>
      <td>2.284781e+01</td>
    </tr>
    <tr>
      <th>avg_length</th>
      <td>2.481495e+02</td>
      <td>2.484051e+02</td>
    </tr>
    <tr>
      <th>stddev(length)</th>
      <td>9.525167e+01</td>
      <td>9.256759e+01</td>
    </tr>
  </tbody>
</table>
</div>




```python

def extract_sample_feature_by_split(df,features, split, split_values):
    """
    allows sampling a spark dataframe without using groupby
    can be used for plotting etc
    """
    
    for n in range(0,len(split_values)):
        values = df.select(features).filter('{} == {}'.format( split, split_values[n])).sample(False, sample_frac).collect()
        numerical_values=[]
        for feature in features:
            column=[x[feature] for x in values]
            numerical_values.append(column)
        feat_col=[split_values[n]] * len(column)
        
        col_dict=dict(zip( features,numerical_values))
        col_dict[split]=feat_col
        nf=pd.DataFrame(col_dict, columns=features+ [split])
        if n==0:
            of=nf.copy()
        else:
            of=of.append(nf, ignore_index=True)
    return(of)
```


```python
features=['different_songs_per_user','different_artists_per_user']
split='churn'
split_values=[0,1]
sample_frac=0.9

df=extract_sample_feature_by_split(user_stats,features, split, split_values)
```


```python
sns.boxplot(x=split, y=features[1],data=df)
```


```python
feature=1

for n in range(0,len(split_values)):
    df[(df.churn==split_values[n])][features[feature]].hist(histtype='step',lw=2, label='{} = {}'.format(split,split_values[n]));
plt.legend(loc=1);
plt.ylabel('count: {}'.format( features[feature] ));
```


```python

fraction=0.2
kdf = ks.from_pandas(user_stats.sample(False, fraction).toPandas())
```


```python
kdf.describe()
```


```python
kdf.groupby('churn').mean()
```


```python
kdf.groupby('churn').std()
```


```python
kdf
```


```python
sns.heatmap( kdf.corr(), annot=True, ftm='.2f'  )
```


```python
fraction=0.2
user_stats.sample(False, fraction).groupBy('churn').agg(
    avg('different_songs_per_user'),avg('total_songs_per_user'),avg('different_artists_per_user')
).toPandas()
```


```python

```


```python


songs_v.filter(user_log.userId==100011).groupBy('userId','song').count().show(50)
# 125, 100011 

# user_log.groupBy('userId').count().orderBy("count",ascending=True).show()
```


```python

```


```python
user_log.groupBy('userId','song').count().groupBy('userId','count').sum().show()
```

## Gender demographics


```python
users_by_gender=user_log.groupBy(['level','gender']).count().toPandas().sort_values(by=['level', 'gender'])
users_by_gender['percentage']=100.*users_by_gender['count']/(user_log.count())

display(users_by_gender)
```


```python
users_by_device=user_log.groupBy(['level','device']).count().toPandas().sort_values(by=['level', 'device'])
users_by_device['percentage']=100.*users_by_device['count']/(user_log.count())

display(users_by_device)
```


```python
# users_by_browser=user_log.groupBy(['level','browser']).count().toPandas().sort_values(by=['level', 'browser'])
# users_by_browser['percentage']=100.*users_by_browser['count']/(user_log.count())

# display(users_by_browser)
```


```python
# user_log.select('browser').dropDuplicates().show()
```

Due to time constrains, I could not go further in the exploration of data.
I will update this post as soon as I have more free time.

