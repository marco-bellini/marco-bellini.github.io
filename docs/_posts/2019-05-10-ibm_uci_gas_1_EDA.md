---
title: "GS 1: Goal and EDA"
permalink: /gas_sensor/gas_eda/
excerpt: "Gas Sensor: Goal and EDA"
last_modified_at: 2020-06-11T15:53:52-04:00
toc: true
toc_label: "Page Index"
toc_sticky: true
author_profile: true
classes: post
tags:
  - EDA
  - Visualization
  - Time Series
  - Deep Learning
categories:
  - Gas_Sensor 
---



# Use Case

The use case is realizing a machine learning algorithm able to infer accurate air quality information (comparable to the prediction of a co-located reference certified analyzer) from data collected by an array of 5 metal oxide chemical sensors embedded in an Air Quality Chemical Multisensor Device. 
The embedded sensor data is less accurate and subject to drift, sensor failure and inaccuracies. The goal is to use machine learning to predict the concentration of Benzene (available only from the reference station).

The business use case is to replicate with an inexpensive, portable sensor the accuracy of a reference certified analyzer.
Further uses of machine learning include the ability to correct inaccuracies in the readings, reconstruct the reference air quality information and predict future readings to detect and possibly compensate for the loss of a sensor.


## Data Set

The dataset is available at UCI (https://archive.ics.uci.edu/ml/datasets/Air+Quality) and is referenced in this works:

[1] S. De Vito, E. Massera, M. Piga, L. Martinotto, G. Di Francia, On field calibration of an electronic nose for benzene estimation in an urban pollution monitoring scenario, Sensors and Actuators B: Chemical, Volume 129, Issue 2, 22 February 2008, Pages 750-757, ISSN 0925-4005, [Web Link].


# Approach

These are relavant points for the described used case
* what kind of output should the model give: 
	- current Benzene concentration at the current time
	- corrected concentration of all gases at the current time 
	- prediction of concentrations at the next time or over a time span
* what is the acceptable error for concentrations at the current time and at the following time steps
* how was the input data collected? Is it possible that the ground truth contains errors?
* what to do if an input signal is not available anymore: raise flag, self-calibrate model


The following steps are taken:
* data quality assessment 
* exploratory data analysis (autocorrelation, seasonality...)
* proposed framework for single and multi-step prediction (single and multiple targets)
* model evaluation at the current and the next step prediction for single target

Next steps:
* detection of faults in sensor input


### Data set information.

As reported in the UCI website, the dataset contains 9358 instances of hourly averaged responses from an array of 5 metal oxide chemical sensors embedded in an Air Quality Chemical Multisensor Device. The device was located on the field in a significantly polluted area, at road level,within an Italian city. Data were recorded from March 2004 to February 2005 (one year)representing the longest freely available recordings of on field deployed air quality chemical sensor devices responses. Ground Truth hourly averaged concentrations for CO, Non Metanic Hydrocarbons, Benzene, Total Nitrogen Oxides (NOx) and Nitrogen Dioxide (NO2) and were provided by a co-located reference certified analyzer. Evidences of cross-sensitivities as well as both concept and sensor drifts are present as described in De Vito et al., Sens. And Act. B, Vol. 129,2,2008 [1] eventually affecting sensors concentration estimation capabilities. 

Missing values are tagged with -200 value. This dataset can be used exclusively for research purposes. Commercial purposes are fully excluded.

Attribute Information:
```
0 Date (DD/MM/YYYY)
1 Time (HH.MM.SS)
2 True hourly averaged concentration CO in mg/m^3 (reference analyzer)
3 PT08.S1 (tin oxide) hourly averaged sensor response (nominally CO targeted)
4 True hourly averaged overall Non Metanic HydroCarbons concentration in microg/m^3 (reference analyzer)
5 True hourly averaged Benzene concentration in microg/m^3 (reference analyzer)
6 PT08.S2 (titania) hourly averaged sensor response (nominally NMHC targeted)
7 True hourly averaged NOx concentration in ppb (reference analyzer)
8 PT08.S3 (tungsten oxide) hourly averaged sensor response (nominally NOx targeted)
9 True hourly averaged NO2 concentration in microg/m^3 (reference analyzer)
10 PT08.S4 (tungsten oxide) hourly averaged sensor response (nominally NO2 targeted)
11 PT08.S5 (indium oxide) hourly averaged sensor response (nominally O3 targeted)
12 Temperature in °C
13 Relative Humidity (%)
14 AH Absolute Humidity
```





## Data Cleansing



### Inital formatting

As shown above we can clean a number of issues at the importing stage:

* the separator is the semicolon
* the decimal separator is the comma as it's typical in Italy, not the decimal point
* there are a number of empty data at the end of the file, that can be skipped since we know the number of valid lines
* the value -200 is used for missing measurements




```python
body = ...
df = pd.read_csv(body,sep=';',skip_blank_lines=True, na_values =[-200],decimal=",", parse_dates=[[0,1]], nrows=9357 )
df.head()                 
                 
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
      <th>Date_Time</th>
      <th>CO(GT)</th>
      <th>PT08.S1(CO)</th>
      <th>NMHC(GT)</th>
      <th>C6H6(GT)</th>
      <th>PT08.S2(NMHC)</th>
      <th>NOx(GT)</th>
      <th>PT08.S3(NOx)</th>
      <th>NO2(GT)</th>
      <th>PT08.S4(NO2)</th>
      <th>PT08.S5(O3)</th>
      <th>T</th>
      <th>RH</th>
      <th>AH</th>
      <th>Unnamed: 15</th>
      <th>Unnamed: 16</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10/03/2004 18.00.00</td>
      <td>2.6</td>
      <td>1360.0</td>
      <td>150.0</td>
      <td>11.9</td>
      <td>1046.0</td>
      <td>166.0</td>
      <td>1056.0</td>
      <td>113.0</td>
      <td>1692.0</td>
      <td>1268.0</td>
      <td>13.6</td>
      <td>48.9</td>
      <td>0.7578</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10/03/2004 19.00.00</td>
      <td>2.0</td>
      <td>1292.0</td>
      <td>112.0</td>
      <td>9.4</td>
      <td>955.0</td>
      <td>103.0</td>
      <td>1174.0</td>
      <td>92.0</td>
      <td>1559.0</td>
      <td>972.0</td>
      <td>13.3</td>
      <td>47.7</td>
      <td>0.7255</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10/03/2004 20.00.00</td>
      <td>2.2</td>
      <td>1402.0</td>
      <td>88.0</td>
      <td>9.0</td>
      <td>939.0</td>
      <td>131.0</td>
      <td>1140.0</td>
      <td>114.0</td>
      <td>1555.0</td>
      <td>1074.0</td>
      <td>11.9</td>
      <td>54.0</td>
      <td>0.7502</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10/03/2004 21.00.00</td>
      <td>2.2</td>
      <td>1376.0</td>
      <td>80.0</td>
      <td>9.2</td>
      <td>948.0</td>
      <td>172.0</td>
      <td>1092.0</td>
      <td>122.0</td>
      <td>1584.0</td>
      <td>1203.0</td>
      <td>11.0</td>
      <td>60.0</td>
      <td>0.7867</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>10/03/2004 22.00.00</td>
      <td>1.6</td>
      <td>1272.0</td>
      <td>51.0</td>
      <td>6.5</td>
      <td>836.0</td>
      <td>131.0</td>
      <td>1205.0</td>
      <td>116.0</td>
      <td>1490.0</td>
      <td>1110.0</td>
      <td>11.2</td>
      <td>59.6</td>
      <td>0.7888</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



```python
df.describe()
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
      <th>CO(GT)</th>
      <th>PT08.S1(CO)</th>
      <th>NMHC(GT)</th>
      <th>C6H6(GT)</th>
      <th>PT08.S2(NMHC)</th>
      <th>NOx(GT)</th>
      <th>PT08.S3(NOx)</th>
      <th>NO2(GT)</th>
      <th>PT08.S4(NO2)</th>
      <th>PT08.S5(O3)</th>
      <th>T</th>
      <th>RH</th>
      <th>AH</th>
      <th>Unnamed: 15</th>
      <th>Unnamed: 16</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>7674.000000</td>
      <td>8991.000000</td>
      <td>914.000000</td>
      <td>8991.000000</td>
      <td>8991.000000</td>
      <td>7718.000000</td>
      <td>8991.000000</td>
      <td>7715.000000</td>
      <td>8991.000000</td>
      <td>8991.000000</td>
      <td>8991.000000</td>
      <td>8991.000000</td>
      <td>8991.000000</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2.152750</td>
      <td>1099.833166</td>
      <td>218.811816</td>
      <td>10.083105</td>
      <td>939.153376</td>
      <td>246.896735</td>
      <td>835.493605</td>
      <td>113.091251</td>
      <td>1456.264598</td>
      <td>1022.906128</td>
      <td>18.317829</td>
      <td>49.234201</td>
      <td>1.025530</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.453252</td>
      <td>217.080037</td>
      <td>204.459921</td>
      <td>7.449820</td>
      <td>266.831429</td>
      <td>212.979168</td>
      <td>256.817320</td>
      <td>48.370108</td>
      <td>346.206794</td>
      <td>398.484288</td>
      <td>8.832116</td>
      <td>17.316892</td>
      <td>0.403813</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.100000</td>
      <td>647.000000</td>
      <td>7.000000</td>
      <td>0.100000</td>
      <td>383.000000</td>
      <td>2.000000</td>
      <td>322.000000</td>
      <td>2.000000</td>
      <td>551.000000</td>
      <td>221.000000</td>
      <td>-1.900000</td>
      <td>9.200000</td>
      <td>0.184700</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1.100000</td>
      <td>937.000000</td>
      <td>67.000000</td>
      <td>4.400000</td>
      <td>734.500000</td>
      <td>98.000000</td>
      <td>658.000000</td>
      <td>78.000000</td>
      <td>1227.000000</td>
      <td>731.500000</td>
      <td>11.800000</td>
      <td>35.800000</td>
      <td>0.736800</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1.800000</td>
      <td>1063.000000</td>
      <td>150.000000</td>
      <td>8.200000</td>
      <td>909.000000</td>
      <td>180.000000</td>
      <td>806.000000</td>
      <td>109.000000</td>
      <td>1463.000000</td>
      <td>963.000000</td>
      <td>17.800000</td>
      <td>49.600000</td>
      <td>0.995400</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2.900000</td>
      <td>1231.000000</td>
      <td>297.000000</td>
      <td>14.000000</td>
      <td>1116.000000</td>
      <td>326.000000</td>
      <td>969.500000</td>
      <td>142.000000</td>
      <td>1674.000000</td>
      <td>1273.500000</td>
      <td>24.400000</td>
      <td>62.500000</td>
      <td>1.313700</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>max</th>
      <td>11.900000</td>
      <td>2040.000000</td>
      <td>1189.000000</td>
      <td>63.700000</td>
      <td>2214.000000</td>
      <td>1479.000000</td>
      <td>2683.000000</td>
      <td>340.000000</td>
      <td>2775.000000</td>
      <td>2523.000000</td>
      <td>44.600000</td>
      <td>88.700000</td>
      <td>2.231000</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



### Data Quality Assessment


We notice:
* the data is a time series of floating point, non negative measurements
* timestep is one hour and the data spans several months
* the ranges are quite different and will require normalization
* there are many missing values




```python
# checking for missing values:

null_values_count=df.isnull().sum()
display(null_values_count)
```


    Date_Time           0
    CO(GT)           1683
    PT08.S1(CO)       366
    NMHC(GT)         8443
    C6H6(GT)          366
    PT08.S2(NMHC)     366
    NOx(GT)          1639
    PT08.S3(NOx)      366
    NO2(GT)          1642
    PT08.S4(NO2)      366
    PT08.S5(O3)       366
    T                 366
    RH                366
    AH                366
    Unnamed: 15      9357
    Unnamed: 16      9357
    dtype: int64


Given the dataset the **method chosen for data quality assessment is the percentage of nan values in the columns**.

Why have I chosen a specific method for data quality assessment?

Time series with large blank intervals cannot be predicted. In our case the ground truth for the NMHC signal is missing 90% of the time. But if one or more of the inputs were missing for large intervals it would be impossible to reconstruct the signal as modelling of time series relies on the relationship between present and past data points.




```python
null_values_count=df.isnull().sum()
perc_missing_values=100.*null_values_count/df.shape[0]
ret=perc_missing_values.sort_values(ascending=False).to_frame()

ret=ret.rename(columns={0:'perc.missing data'})

display(ret)
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
      <th>perc.missing data</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Unnamed: 16</th>
      <td>100.000000</td>
    </tr>
    <tr>
      <th>Unnamed: 15</th>
      <td>100.000000</td>
    </tr>
    <tr>
      <th>NMHC(GT)</th>
      <td>90.231912</td>
    </tr>
    <tr>
      <th>CO(GT)</th>
      <td>17.986534</td>
    </tr>
    <tr>
      <th>NO2(GT)</th>
      <td>17.548360</td>
    </tr>
    <tr>
      <th>NOx(GT)</th>
      <td>17.516298</td>
    </tr>
    <tr>
      <th>AH</th>
      <td>3.911510</td>
    </tr>
    <tr>
      <th>RH</th>
      <td>3.911510</td>
    </tr>
    <tr>
      <th>T</th>
      <td>3.911510</td>
    </tr>
    <tr>
      <th>PT08.S5(O3)</th>
      <td>3.911510</td>
    </tr>
    <tr>
      <th>PT08.S4(NO2)</th>
      <td>3.911510</td>
    </tr>
    <tr>
      <th>PT08.S3(NOx)</th>
      <td>3.911510</td>
    </tr>
    <tr>
      <th>PT08.S2(NMHC)</th>
      <td>3.911510</td>
    </tr>
    <tr>
      <th>C6H6(GT)</th>
      <td>3.911510</td>
    </tr>
    <tr>
      <th>PT08.S1(CO)</th>
      <td>3.911510</td>
    </tr>
    <tr>
      <th>Date_Time</th>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>



```python
# we drop column 15 and 16 since they are clearly empty

df=df.drop(columns=['Unnamed: 15','Unnamed: 16',])

# we create a datetime object that could be used as an index
df['dt']=pd.to_datetime(df['Date_Time'],dayfirst =True,format='%d/%m/%Y %H.%M.%S')

df.head()
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
      <th>Date_Time</th>
      <th>CO(GT)</th>
      <th>PT08.S1(CO)</th>
      <th>NMHC(GT)</th>
      <th>C6H6(GT)</th>
      <th>PT08.S2(NMHC)</th>
      <th>NOx(GT)</th>
      <th>PT08.S3(NOx)</th>
      <th>NO2(GT)</th>
      <th>PT08.S4(NO2)</th>
      <th>PT08.S5(O3)</th>
      <th>T</th>
      <th>RH</th>
      <th>AH</th>
      <th>dt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10/03/2004 18.00.00</td>
      <td>2.6</td>
      <td>1360.0</td>
      <td>150.0</td>
      <td>11.9</td>
      <td>1046.0</td>
      <td>166.0</td>
      <td>1056.0</td>
      <td>113.0</td>
      <td>1692.0</td>
      <td>1268.0</td>
      <td>13.6</td>
      <td>48.9</td>
      <td>0.7578</td>
      <td>2004-03-10 18:00:00</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10/03/2004 19.00.00</td>
      <td>2.0</td>
      <td>1292.0</td>
      <td>112.0</td>
      <td>9.4</td>
      <td>955.0</td>
      <td>103.0</td>
      <td>1174.0</td>
      <td>92.0</td>
      <td>1559.0</td>
      <td>972.0</td>
      <td>13.3</td>
      <td>47.7</td>
      <td>0.7255</td>
      <td>2004-03-10 19:00:00</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10/03/2004 20.00.00</td>
      <td>2.2</td>
      <td>1402.0</td>
      <td>88.0</td>
      <td>9.0</td>
      <td>939.0</td>
      <td>131.0</td>
      <td>1140.0</td>
      <td>114.0</td>
      <td>1555.0</td>
      <td>1074.0</td>
      <td>11.9</td>
      <td>54.0</td>
      <td>0.7502</td>
      <td>2004-03-10 20:00:00</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10/03/2004 21.00.00</td>
      <td>2.2</td>
      <td>1376.0</td>
      <td>80.0</td>
      <td>9.2</td>
      <td>948.0</td>
      <td>172.0</td>
      <td>1092.0</td>
      <td>122.0</td>
      <td>1584.0</td>
      <td>1203.0</td>
      <td>11.0</td>
      <td>60.0</td>
      <td>0.7867</td>
      <td>2004-03-10 21:00:00</td>
    </tr>
    <tr>
      <th>4</th>
      <td>10/03/2004 22.00.00</td>
      <td>1.6</td>
      <td>1272.0</td>
      <td>51.0</td>
      <td>6.5</td>
      <td>836.0</td>
      <td>131.0</td>
      <td>1205.0</td>
      <td>116.0</td>
      <td>1490.0</td>
      <td>1110.0</td>
      <td>11.2</td>
      <td>59.6</td>
      <td>0.7888</td>
      <td>2004-03-10 22:00:00</td>
    </tr>
  </tbody>
</table>
</div>




```python
# we rename the sensor columns for better clarity

gt_columns=[x for x in list(df.columns) if '(GT)' in x ]
sensor_columns=[x for x in list(df.columns) if '.S' in x ]
sensor_columns_renamed=[x.split('(')[1].split(')')[0] for x in list(df.columns) if '.S' in x ]

df=df.rename(columns=dict(zip(sensor_columns,sensor_columns_renamed) ) )
```


```python
# the data is now much easier to read and plot

df.head()
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
      <th>Date_Time</th>
      <th>CO(GT)</th>
      <th>CO</th>
      <th>NMHC(GT)</th>
      <th>C6H6(GT)</th>
      <th>NMHC</th>
      <th>NOx(GT)</th>
      <th>NOx</th>
      <th>NO2(GT)</th>
      <th>NO2</th>
      <th>O3</th>
      <th>T</th>
      <th>RH</th>
      <th>AH</th>
      <th>dt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10/03/2004 18.00.00</td>
      <td>2.6</td>
      <td>1360.0</td>
      <td>150.0</td>
      <td>11.9</td>
      <td>1046.0</td>
      <td>166.0</td>
      <td>1056.0</td>
      <td>113.0</td>
      <td>1692.0</td>
      <td>1268.0</td>
      <td>13.6</td>
      <td>48.9</td>
      <td>0.7578</td>
      <td>2004-03-10 18:00:00</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10/03/2004 19.00.00</td>
      <td>2.0</td>
      <td>1292.0</td>
      <td>112.0</td>
      <td>9.4</td>
      <td>955.0</td>
      <td>103.0</td>
      <td>1174.0</td>
      <td>92.0</td>
      <td>1559.0</td>
      <td>972.0</td>
      <td>13.3</td>
      <td>47.7</td>
      <td>0.7255</td>
      <td>2004-03-10 19:00:00</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10/03/2004 20.00.00</td>
      <td>2.2</td>
      <td>1402.0</td>
      <td>88.0</td>
      <td>9.0</td>
      <td>939.0</td>
      <td>131.0</td>
      <td>1140.0</td>
      <td>114.0</td>
      <td>1555.0</td>
      <td>1074.0</td>
      <td>11.9</td>
      <td>54.0</td>
      <td>0.7502</td>
      <td>2004-03-10 20:00:00</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10/03/2004 21.00.00</td>
      <td>2.2</td>
      <td>1376.0</td>
      <td>80.0</td>
      <td>9.2</td>
      <td>948.0</td>
      <td>172.0</td>
      <td>1092.0</td>
      <td>122.0</td>
      <td>1584.0</td>
      <td>1203.0</td>
      <td>11.0</td>
      <td>60.0</td>
      <td>0.7867</td>
      <td>2004-03-10 21:00:00</td>
    </tr>
    <tr>
      <th>4</th>
      <td>10/03/2004 22.00.00</td>
      <td>1.6</td>
      <td>1272.0</td>
      <td>51.0</td>
      <td>6.5</td>
      <td>836.0</td>
      <td>131.0</td>
      <td>1205.0</td>
      <td>116.0</td>
      <td>1490.0</td>
      <td>1110.0</td>
      <td>11.2</td>
      <td>59.6</td>
      <td>0.7888</td>
      <td>2004-03-10 22:00:00</td>
    </tr>
  </tbody>
</table>
</div>



### Data Visualization

Since the data is a timeseries we plot it before the data exploration phase, as it makes little sense to look blindly for correlations


```python
# initial plot

fig, ax = plt.subplots(nrows=4,sharex=True)
fig.set_size_inches(14,7)


# we copy the database and reindex using the dt column in datetime format to show the month and day
df2=df.copy()
df2=df2.set_index(['dt'])
df2=df2.drop(columns=['Date_Time'])

n=0
for gas in ['CO','NMHC','NOx','NO2']:
    df2[gas].plot(ax=ax[n],color='darkred',alpha=0.5)
    df2[gas+'(GT)'].plot(ax=ax[n],color='steelblue',alpha=0.5)
    ax[n].legend(loc=1)
    n+=1

plt.suptitle('Sensor data vs Ground Truth');
```


{% raw %}![alt](/assets/ibm_gas_sensor/ts_unscaled.png){% endraw %}

The plot confirms:
* need to rescale the data as the sensors and reference signals are not identical (probably because of different systems or data scale etc...)
* large amount of missing values
* the NMHC certified analyzer stops operating  in May 2004




```python
# additional sensor data

fig, ax = plt.subplots(nrows=3,sharex=True)
fig.set_size_inches(14,5)

n=0
for gas in ['T','RH','AH']:
    df2[gas].plot(ax=ax[n],alpha=0.5)
    ax[n].legend(loc=1)
    n+=1

plt.suptitle('Sensor data (Temperature, Absolute and Relative Humidity)');
```

{% raw %}![alt](/assets/ibm_gas_sensor/ts_humidity.png){% endraw %}




```python
# additional sensor data

fig, ax = plt.subplots(nrows=1,sharex=True)
fig.set_size_inches(14,3)

df2['C6H6(GT)'].plot(ax=ax,alpha=0.5, color='darkgreen')
ax.legend(loc=1)

plt.suptitle('Target data (C6H6, Benzene)');
```

{% raw %}![alt](/assets/ibm_gas_sensor/ts_target.png){% endraw %}



### Data Exploration 

We normalize each of them individually, to properly compare the signals.


```python
def df_normalize(df):
    mean=df.mean()
    std=df.std()
    df2=df.copy()
    df2=(df2-mean)/std
    return df2,mean,std

df3,mean,std=df_normalize(df2)
```


```python
fig, ax = plt.subplots(nrows=4,sharex=True)
fig.set_size_inches(14,7)

n=0
for gas in ['CO','NMHC','NOx','NO2']:
    df3[gas].plot(ax=ax[n],color='darkred',alpha=0.5)
    df3[gas+'(GT)'].plot(ax=ax[n],color='steelblue',alpha=0.5)
    ax[n].legend(loc=1)
    n+=1

plt.suptitle('Normalized Sensor data vs Ground Truth');
```

{% raw %}![alt](/assets/ibm_gas_sensor/ts_scaled.png){% endraw %}


The normalized data shows clear trends, therefore the **Feature Engineering** step that we will perform will be **Normalization**.

As for data quality we notice no obvious red flags such as:
* sensor stuck on some values
* sensors saturating at high or low values


#### Cross-correlation

We see that at the embedded sensor show similar trends to the reference analyzer, however some work is needed to reconstruct the ground truth.

The proper way to assess correlations is to use statsmodels cross-correlation function for time series, after dropping the nan values.


https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.ccf.html


```python
fig, ax = plt.subplots(nrows=1,sharex=True)
fig.set_size_inches(8,4)

n=0
for gas in ['CO','NMHC','NOx','NO2']:
    # drop the nan to avoid errors
    cmp=df3[[gas,gas+'(GT)']].dropna(axis=0)
    ts_corr=ts.stattools.ccf(cmp[gas],cmp[gas+'(GT)'])
    cmp_index=cmp.index
    ts_corr=pd.DataFrame(ts_corr)#,index=cmp_index)

    plt.plot(ts_corr,label=gas,alpha=0.5)

plt.legend()
plt.xlabel('hours');
plt.title('correlations between Sensor and GT for each gas');

# we limit the analyis to a lag of 1 week
plt.xlim(0,7*24);
```

{% raw %}![alt](/assets/ibm_gas_sensor/corr.png){% endraw %}



As expected there is correlation between sensors and reference. Negative correlation may be due to the response type of the sensor that needs post-processing.


```python
fig, ax = plt.subplots(nrows=1,sharex=True)
fig.set_size_inches(8,4)

n=0
for gas in ['NMHC','NOx','NO2']:
    # drop the nan to avoid errors
    cmp=df3[['CO',gas]].dropna(axis=0)
    ts_corr=ts.stattools.ccf(cmp['CO'],cmp[gas])
    cmp_index=cmp.index
    ts_corr=pd.DataFrame(ts_corr)#,index=cmp_index)

    plt.plot(ts_corr,label=gas,alpha=0.5)

plt.legend()
plt.xlabel('hours');
plt.title('correlations between CO and other gases');

# we limit the analyis to a lag of 1 week
plt.xlim(0,7*24);
```


{% raw %}![alt](/assets/ibm_gas_sensor/corr2.png){% endraw %}



Interestingly there seems to be a strong crosscorrelation between the reading.


```python
fig, ax = plt.subplots(nrows=1,sharex=True)
fig.set_size_inches(8,4)

n=0
for gas in ['CO','NMHC','NOx','NO2','T','RH','AH']:
    # drop the nan to avoid errors
    cmp=df3[['C6H6(GT)',gas]].dropna(axis=0)
    ts_corr=ts.stattools.ccf(cmp['C6H6(GT)'],cmp[gas])
    cmp_index=cmp.index
    ts_corr=pd.DataFrame(ts_corr)#,index=cmp_index)

    plt.plot(ts_corr,label=gas,alpha=0.5)

plt.legend(loc=4)
plt.xlabel('hours');
plt.title('correlations between the target (C6H6) and the sensor readings');

# we limit the analyis to a lag of 1 week
plt.xlim(0,7*24);
plt.ylim(-1,1);

```

{% raw %}![alt](/assets/ibm_gas_sensor/corr3.png){% endraw %}


The high level of correlation inspires confidence in the approach.


#### Seasonality 




```python
yBenz=df3['C6H6(GT)']

fig=plt.figure(figsize=(14,8))
ax1=plt.subplot(211)
fig2=tsaplots.plot_acf(yBenz.dropna(), lags=200, ax=ax1 )

ax2=plt.subplot(212)
fig3=tsaplots.plot_pacf(yBenz.dropna(), lags=200, ax=ax2 )
plt.suptitle('C6H6 Benzene (target)');

```

{% raw %}![alt](/assets/ibm_gas_sensor/autocorr.png){% endraw %}



We note that the gas concentrations follow a daily pattern. A data window >= 24 is advised.


```python
# longer trends

fig=plt.figure(figsize=(14,4))
plt.plot(yBenz.dropna(),alpha=0.2, label='Y' )
plt.plot(yBenz.resample('D').mean(), label='Y daily' )
plt.plot(yBenz.resample('W').mean(), label='Y weekly' )
plt.legend(loc=0);

```

{% raw %}![alt](/assets/ibm_gas_sensor/trends.png){% endraw %}



The daily tred is confirmed. There seems to be a weekly trend but only a very weak monthly trend.



