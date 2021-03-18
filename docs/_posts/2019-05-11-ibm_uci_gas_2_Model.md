---
title: "GS 2: Model"
permalink: /gas_sensor/gas_model/
excerpt: "Gas Sensor: Modeling"
last_modified_at: 2020-06-12T15:53:52-04:00
toc: true
toc_label: "Page Index"
toc_sticky: true
author_profile: true
classes: post
tags:
  - Time Series
  - Neural Networks
  - LSTM
categories:
  - Gas_Sensor 
---


We reload the data and apply basic transformations as explained before.


```python

df = pd.read_csv(body,sep=';',skip_blank_lines=True, na_values =[-200],decimal=",", parse_dates=[[0,1]], nrows=9357 )

# we drop column 15 and 16 since they are clearly empty

df=df.drop(columns=['Unnamed: 15','Unnamed: 16',])

# we create a datetime object that could be used as an index
df['dt']=pd.to_datetime(df['Date_Time'],dayfirst =True,format='%d/%m/%Y %H.%M.%S')

gt_columns=[x for x in list(df.columns) if '(GT)' in x ]
sensor_columns=[x for x in list(df.columns) if '.S' in x ]
sensor_columns_renamed=[x.split('(')[1].split(')')[0] for x in list(df.columns) if '.S' in x ]

df=df.rename(columns=dict(zip(sensor_columns,sensor_columns_renamed) ) )
```


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


{% raw %}![alt](/assets/ibm_gas_sensor/starting_point.png){% endraw %}

We normalize each feature.

```python
def df_normalize(df):
    mean=df.mean()
    std=df.std()
    df2=df.copy()
    df2=(df2-mean)/std
    return df2,mean,std

df3,mean,std=df_normalize(df2)
```

# Machine Learning Model


## Model and framework choice

The goal is to reconstruct the benzene reference signal `C6H6(GT)` from the embedded sensor data.
While complex models such as [SARIMAX](https://www.statsmodels.org/dev/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html) could be used, we limit ourselves to a simple linear regression model, to stay within the material covered in the course. 

We split the dataset in two chronological chunks, with the first 6000 datapoints as the training set.


First we define a basic, simple baseline for prediction usign Linear Regression, using scikit-learn. 

```python
df_clean=df2[['CO', 'NMHC', 'NOx', 'NO2', 'O3', 'T', 'RH', 'AH', 'C6H6(GT)']].dropna(how='any')
print('non empty timesteps: ',len(df_clean))

Xtrain=df_clean[0:6000].copy()
Ytrain=Xtrain.pop('C6H6(GT)')

Xtest=df_clean[6000:].copy()
Ytest=Xtest.pop('C6H6(GT)')
```

    non empty timesteps:  8991
    


```python
from sklearn.linear_model import LinearRegression

reg = LinearRegression().fit(Xtrain, Ytrain)

Ytrain_pred=reg.predict(Xtrain)
Ytrain_pred_plot=pd.DataFrame(Ytrain_pred,index=Ytrain.index,columns=['Ytrain_pred'])

fig, ax = plt.subplots(nrows=2,sharex=False)
fig.set_size_inches(14,10)

Ytrain.plot(ax=ax[0],   alpha=0.5);
Ytrain_pred_plot.plot(ax=ax[0], color='darkred',  alpha=0.5)
ax[0].legend(loc=1)
ax[0].set_title('Training set');


Ytest_pred=reg.predict(Xtest)
Ytest_pred_plot=pd.DataFrame(Ytest_pred,index=Ytest.index,columns=['Ytest_pred'])


Ytest.plot(ax=ax[1],  alpha=0.5);
Ytest_pred_plot.plot(ax=ax[1], color='darkred', alpha=0.5)
ax[1].legend(loc=1)
ax[1].set_title('Test set');

```

{% raw %}![alt](/assets/ibm_gas_sensor/output_9_0.png){% endraw %}



## Model Performance Indicator

The metric chosen to asses performance is the **Mean Absolute Error** (MAE) as:
* it is less sensitive to outliers
* it is very simple to interpret


```python
from sklearn.metrics import mean_absolute_error,  mean_squared_error
print('Training:')
print('MAE: ',mean_absolute_error(Ytrain.values, Ytrain_pred))
print('MSE: ',mean_squared_error(Ytrain.values, Ytrain_pred))
print('Max Abs error: ',  np.max(np.abs(Ytrain.values- Ytrain_pred)))

print()
print('Test:')
print('MAE: ',mean_absolute_error(Ytest.values, Ytest_pred))
print('MSE: ',mean_squared_error(Ytest.values, Ytest_pred))
print('Max Abs error: ',  np.max(np.abs(Ytest.values- Ytest_pred)))


    Training:
    MAE:  0.7306005527999552
    MSE:  1.0261894399497948
    Max Abs error:  8.911949104521561
    
    Test:
    MAE:  1.8115724785475074
    MSE:  5.122512365727219
    Max Abs error:  13.79365752667091
    
```

```python
err=Ytest.values-Ytest_pred
err=pd.DataFrame(err,index=Ytest.index,columns=['err'])

fig, ax = plt.subplots(nrows=1,sharex=False)
fig.set_size_inches(14,3)
err.plot(ax=ax)
plt.title('Prediction error on test set');
```

{% raw %}![alt](/assets/ibm_gas_sensor/output_12_0.png){% endraw %}


```python
# Regression coefficients

feat_imp=pd.DataFrame(reg.coef_.reshape(1,-1), columns=Xtrain.columns, index=[0])
display(feat_imp)
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
      <th>CO</th>
      <th>NMHC</th>
      <th>NOx</th>
      <th>NO2</th>
      <th>O3</th>
      <th>T</th>
      <th>RH</th>
      <th>AH</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.001188</td>
      <td>0.034988</td>
      <td>0.008561</td>
      <td>-0.000841</td>
      <td>0.001038</td>
      <td>-0.038385</td>
      <td>0.002842</td>
      <td>2.238023</td>
    </tr>
  </tbody>
</table>
</div>



# Deep Learning Model


## Model and framework choice

The goal is to reconstruct the benzene reference signal `C6H6(GT)` from the embedded sensor data.
In this section we choose a more appropriate algorithm for timeseries, i.e. `LSTM` using tensorflow 2.2.

The reason for choosing this specific algorithm is that LSTM are a Recurrent Neural Network architecture that is suited to model sequences.
Alternative architectures to be considered are GRU.




```python
def multivariate_data_s_gt(dataset_s_pd,dataset_gt_pd, start_index, end_index, history_size, target_size,multi_step=False, 
                         return_sensor_plt=False, include_sensor_present=False):
    """
    this function has several purposes:
    - 1. extract sequences Xtrain and Ytrain of arbitrary length (history_size) from the subset start_index:end_index of the X (dataset_s_pd) and Y (dataset_s_pd) datasets
    - 2. extract the target data as a single point (prediction of the current Y step) with (target_size=0) or with future steps (target_size>0) 
         the sensor current step can be included or not (to simulate a lag etc...)
    - 3. extract sequences used for debugging and plotting X(sensor_plt) and Y(sensor_y) for illustration purposes
        in this case return_sensor_plt=True,  multi_step as appropriate 
        
    The goal of the function is to permit analysis beyond the scope of the capstone, e.g.:
    - detect and address faults in sensors...
    - multi-step prediction of Benzene concentration 
        
    TODO:
    - improve naming of variables
    
    """ 
    
    
    
    # dataset_s_pd is the sensor data (gases, T, RH, AH)
    # X
    dataset_s=dataset_s_pd.values
    
    # dataset_gt is the ground truth (e.g. Benzene) 
    # Y
    dataset_gt=dataset_gt_pd.values
    
    data = []
    labels = []
    
    # candidates for the plotting 3.
    sensor_plt=[]
    cand_y_plt=[]
    sensor_y=[]
    
    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset_s) - target_size

    for i in range(start_index, end_index):
        if not include_sensor_present:
            # include the current time step
            indices = range(i-history_size, i)
            # Reshape data from (history_size,) to (history_size, 1)
            cand_data=dataset_s[indices]
            cand_y_plt=dataset_gt[indices]
        else:
            indices = range(i-history_size, i+1)
            cand_data=dataset_s[indices]
            cand_y_plt=dataset_gt[indices]

        # plotting data
        if multi_step:
            cand_label=dataset_gt[i:i+target_size]
            # just for easy plotting of the comparable sensor data
            cand_sensor_plt=dataset_s[i:i+target_size]
        else:
            cand_label=dataset_gt[i+target_size]
            cand_sensor_plt=dataset_s[i+target_size]
            
        if not( np.isnan(cand_data).any() | np.isnan(cand_label).any() ):
            data.append(cand_data)
            labels.append(cand_label)
            sensor_plt.append(cand_sensor_plt)
            sensor_y.append(cand_y_plt)

    sensor_plt=np.array(sensor_plt)
    sensor_y=np.array(sensor_y)
        
    if not return_sensor_plt:
        return np.array(data), np.array(labels)
    else:
        return np.array(data), np.array(labels),sensor_plt,sensor_y
    
```


```python
# we use 5000 datapoints for training and the remaining 2000+ for the testing
TRAIN_SPLIT = 5000

# each sequence is made by 48 points (i.e. 2 days)
multivariate_past_history = 48
# the target window is the current Benzene data
multivariate_future_target = 0

# we predict the benzene at time t, with the sensor input up to t-1
include_sensor_present=False

Xcolumns=[  'CO',  'NMHC', 'NOx',  'NO2', 'O3', 'T', 'RH', 'AH']
Ycolumns='C6H6(GT)'

Xsensors=df3[Xcolumns]
Ytrue=df3[Ycolumns]

# training set
x_train_uni, y_train_uni , y_train_sens,y_train_y= multivariate_data_s_gt(Xsensors,Ytrue, 0, TRAIN_SPLIT,
                                           multivariate_past_history,
                                           multivariate_future_target,return_sensor_plt=True, include_sensor_present=include_sensor_present)
# testing set
x_val_uni, y_val_uni , y_val_sens, y_val_y= multivariate_data_s_gt(Xsensors,Ytrue, TRAIN_SPLIT, None,
                                       multivariate_past_history,
                                       multivariate_future_target,return_sensor_plt=True, include_sensor_present=include_sensor_present)

```


```python
def plot_sensors( x_train_uni,y_train_sens, training_step, Xcolumns, include_sensor_present=False ):
    # plots the signals batch (X) and their next value
    # TODO: improve naming conventions

    n_sets, n_window, n_signals=x_train_uni.shape
    
    # x values for plot
    ts=np.arange(-n_window,0)+training_step
    for n in range(0,n_signals):
        hh,=plt.plot(ts,x_train_uni[training_step,:,n], label=Xcolumns[n]);
        
        # sensor present
        if include_sensor_present:
            t_shift=0
        else:
            t_shift=1
        plt.plot([ts[-1]+t_shift],y_train_sens[training_step][n],'X',color=hh.get_color());
    return
```


```python


fig=plt.figure(figsize=(18,4))

plt.subplot(131)
plot_sensors( x_train_uni,y_train_sens, 0, Xcolumns )
plt.legend(loc=2)
plt.title('1st batch')
plt.xlabel('batch timesteps')
plt.ylabel('Sensor data (norm.)')

plt.subplot(132)
plot_sensors( x_train_uni,y_train_sens, 1, Xcolumns )
plt.title('2nd batch')
plt.xlabel('batch timesteps')

plt.subplot(133)
plot_sensors( x_train_uni,y_train_sens, 2, Xcolumns )
plt.title('3rd batch')
plt.xlabel('batch timesteps')

plt.suptitle('Sensor inputs for the first three training batches');
```

{% raw %}![alt](/assets/ibm_gas_sensor/output_24_0.png){% endraw %}



The figure above shows the first three training batches consisting of the sensor input (colored lines). The present values at times 0,1,2 is indicated by a cross and it is shown only for reference purposes.

We will propose a network that predicts only the benzene value (not shown in this graph, obviously) and not the expected future value of the sensors, although this could be a very interesting additional develpment.

Also in this analysis the **present values of the sensors are not used** intentionally to see if the neural network can still provide a reasonable prediction. If this is possible, we could pair two networks, one with and one without access to present values and compare their predictions. Large and unjustified discrepancies between the predictions of the two networks could be used to detect a fault in the sensors. 



```python
def plot_target( y_train_uni,y_train_y, training_step, y_pred=None,y_pred_col='r',  **kwargs ):
    # plots the signals batch (X) and their next value

    n_sets, n_window=y_train_y.shape
    
    # x values for plot
    ts=np.arange(-n_window,0)+training_step

    hh,=plt.plot(ts,y_train_y[training_step], **kwargs);
    plt.plot([ts[-1]+1],y_train_uni[training_step],'o',color=hh.get_color());

    if not y_pred is None:
        plt.plot([ts[-1]+1],y_pred,'x',color=y_pred_col);

    
    return
```


```python


fig=plt.figure(figsize=(18,4))

plt.subplot(131)
plot_target( y_train_uni,y_train_y, 0, label='C6H6')
plt.legend(loc=2)
plt.title('1st batch')
plt.xlabel('batch timesteps')
plt.ylabel('Benzene data (norm.)')


plt.subplot(132)
plot_target( y_train_uni,y_train_y, 1, label='C6H6')
plt.xlabel('batch timesteps')
plt.title('2nd batch')

plt.subplot(133)
plot_target( y_train_uni,y_train_y, 2, label='C6H6')
plt.xlabel('batch timesteps')
plt.title('3rd batch')

plt.suptitle('Reference target for the first three training batches');
```

{% raw %}![alt](/assets/ibm_gas_sensor/output_27_0.png){% endraw %}


The figure above shows the target prediction value for first three training batches (round symbol only). 
The past values are indicated by a solid line and are present for illustration purposes only.



```python
# create batches for efficient training with TF Datasets

BATCH_SIZE = 48*6
BUFFER_SIZE = 10000

train_univariate = tf.data.Dataset.from_tensor_slices((x_train_uni, y_train_uni))
train_univariate = train_univariate.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

val_univariate = tf.data.Dataset.from_tensor_slices((x_val_uni, y_val_uni))
val_univariate = val_univariate.batch(BATCH_SIZE).repeat()

# val_sensor = tf.data.Dataset.from_tensor_slices((y_val_uni, y_val_sens))
# val_sensor = val_sensor.batch(BATCH_SIZE).repeat()
```


```python
simple_lstm_model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(4,dropout=0.2, input_shape=x_train_uni.shape[-2:], return_sequences=False,),
    tf.keras.layers.Dense(1)
])

simple_lstm_model.compile(optimizer='adam', loss='mae')
```


```python
# sanity check

for x, y in val_univariate.take(1):
    print(simple_lstm_model.predict(x).shape)
    
```

    (288, 1)
    


```python
EarlyStopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

# tensorboard
# log_dir=os.path.join(os.getcwd() , r"logs\fit",  datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
# print(log_dir)
# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

```


```python
EVALUATION_INTERVAL = 200*2
EPOCHS = 10

history=simple_lstm_model.fit(train_univariate, epochs=EPOCHS,
                     steps_per_epoch=EVALUATION_INTERVAL,
                      validation_data=val_univariate, validation_steps=50,
                             callbacks=[EarlyStopping]
                              #,tensorboard_callback]
                             )
```

    Epoch 1/10
    400/400 [==============================] - 33s 82ms/step - loss: 0.4867 - val_loss: 0.6038
    Epoch 2/10
    400/400 [==============================] - 29s 74ms/step - loss: 0.3830 - val_loss: 0.5048
    Epoch 3/10
    400/400 [==============================] - 29s 74ms/step - loss: 0.3646 - val_loss: 0.4700
    Epoch 4/10
    400/400 [==============================] - 30s 74ms/step - loss: 0.3558 - val_loss: 0.4391
    Epoch 5/10
    400/400 [==============================] - 29s 72ms/step - loss: 0.3490 - val_loss: 0.4230
    Epoch 6/10
    400/400 [==============================] - 28s 70ms/step - loss: 0.3447 - val_loss: 0.4140
    Epoch 7/10
    400/400 [==============================] - 29s 71ms/step - loss: 0.3408 - val_loss: 0.4103
    Epoch 8/10
    400/400 [==============================] - 28s 70ms/step - loss: 0.3391 - val_loss: 0.4202
    Epoch 9/10
    400/400 [==============================] - 28s 70ms/step - loss: 0.3355 - val_loss: 0.4146
    Epoch 10/10
    400/400 [==============================] - 28s 69ms/step - loss: 0.3346 - val_loss: 0.4114
    

### Evaluation of results


```python
# visualize the error at step n

n=0

x=x_val_uni[n,:,:]
x=x[np.newaxis,:,:]
x.shape
y_pred=simple_lstm_model(x)

plot_target( y_val_uni,y_val_y, n,y_pred=y_pred , label='C6H6')
plt.xlabel('batch timesteps')
plt.ylabel('Benzene data (norm.)')
```





{% raw %}![alt](/assets/ibm_gas_sensor/output_35_1.png){% endraw %}

![png](output_35_1.png)



```python
# calculate the predictions for the whole validation set

y_pred=[]
for n in progressbar.progressbar(range(0,x_val_uni.shape[0]) ):
    x=x_val_uni[n,:,:]
    x=x[np.newaxis,:,:]
    y_pred.append(simple_lstm_model(x))

```

    100% (3803 of 3803) |####################| Elapsed Time: 0:03:25 Time:  0:03:25
    


```python
# reformat for easier visualization
y_pred_n=np.array([x.numpy()[0][0] for x in y_pred])
```


```python
# compare y_pred and ground truth

fig=plt.figure(figsize=(12,4))
plt.plot(y_pred_n,label='y_pred',alpha=0.5);
plt.plot(y_val_uni,label='y_true',alpha=0.5);
plt.legend(loc=0)
plt.xlabel('time step')
plt.ylabel('C6H6 (norm)');
```

{% raw %}![alt](/assets/ibm_gas_sensor/output_38_0.png){% endraw %}



```python
# plot of the error
fig=plt.figure(figsize=(12,4))
plt.plot(y_val_uni-y_pred_n,label='error',alpha=0.5);
plt.legend(loc=0)
plt.xlabel('time step')
plt.ylabel('prediction error');
```

{% raw %}![alt](/assets/ibm_gas_sensor/output_39_0.png){% endraw %}



```python
# we compute the same metrics as for the Linear Regression 

print('Test:')
print('MAE: ',mean_absolute_error(y_val_uni, y_pred_n))
print('MSE: ',mean_squared_error(y_val_uni, y_pred_n))
print('Max Abs error: ',  np.max(np.abs(y_val_uni- y_pred_n)))

    Test:
    MAE:  0.40577815789672306
    MSE:  0.3694934933127622
    Max Abs error:  5.046353151576748

```

    

Not surprisingly we notice that the prediction error is much smaller for the LSTM


### Model including the present value of the sensors


```python
# we use 5000 datapoints for training and the remaining 2000+ for the testing
TRAIN_SPLIT = 5000

# each sequence is made by 48 points (i.e. 2 days)
multivariate_past_history = 48
# the target window is the current Benzene data
multivariate_future_target = 0

# we predict the benzene at time t, with the sensor input up to t-1
include_sensor_present=True

Xcolumns=[  'CO',  'NMHC', 'NOx',  'NO2', 'O3', 'T', 'RH', 'AH']
Ycolumns='C6H6(GT)'

Xsensors=df3[Xcolumns]
Ytrue=df3[Ycolumns]

# training set
x_train_uni, y_train_uni , y_train_sens,y_train_y= multivariate_data_s_gt(Xsensors,Ytrue, 0, TRAIN_SPLIT,
                                           multivariate_past_history,
                                           multivariate_future_target,return_sensor_plt=True, include_sensor_present=include_sensor_present)
# testing set
x_val_uni, y_val_uni , y_val_sens, y_val_y= multivariate_data_s_gt(Xsensors,Ytrue, TRAIN_SPLIT, None,
                                       multivariate_past_history,
                                       multivariate_future_target,return_sensor_plt=True, include_sensor_present=include_sensor_present)

```


```python
fig=plt.figure(figsize=(18,4))

plt.subplot(131)
plot_sensors( x_train_uni,y_train_sens, 0, Xcolumns, include_sensor_present=True)
plt.legend(loc=2)
plt.title('1st batch')
plt.xlabel('batch timesteps')
plt.ylabel('Sensor data (norm.)')

plt.subplot(132)
plot_sensors( x_train_uni,y_train_sens, 1, Xcolumns, include_sensor_present=True )
plt.title('2nd batch')
plt.xlabel('batch timesteps')

plt.subplot(133)
plot_sensors( x_train_uni,y_train_sens, 2, Xcolumns, include_sensor_present=True )
plt.title('3rd batch')
plt.xlabel('batch timesteps')

plt.suptitle('Sensor inputs for the first three training batches');
```

{% raw %}![alt](/assets/ibm_gas_sensor/output_43_0.png){% endraw %}


This time the input data contains the sensors' current status.


```python
# create batches for efficient training with TF Datasets

BATCH_SIZE = 48*6
BUFFER_SIZE = 10000

train_univariate = tf.data.Dataset.from_tensor_slices((x_train_uni, y_train_uni))
train_univariate = train_univariate.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

val_univariate = tf.data.Dataset.from_tensor_slices((x_val_uni, y_val_uni))
val_univariate = val_univariate.batch(BATCH_SIZE).repeat()
```


```python
simple_lstm_model2 = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(4,dropout=0.2, input_shape=x_train_uni.shape[-2:], return_sequences=False,),
    tf.keras.layers.Dense(1)
])

simple_lstm_model2.compile(optimizer='adam', loss='mae')
```


```python
EVALUATION_INTERVAL = 200*2
EPOCHS = 10

history=simple_lstm_model2.fit(train_univariate, epochs=EPOCHS,
                     steps_per_epoch=EVALUATION_INTERVAL,
                      validation_data=val_univariate, validation_steps=50,
                             callbacks=[EarlyStopping]
                              #,tensorboard_callback]
                             )
```

    Epoch 1/10
    400/400 [==============================] - 33s 81ms/step - loss: 0.3504 - val_loss: 0.3210
    Epoch 2/10
    400/400 [==============================] - 28s 71ms/step - loss: 0.1922 - val_loss: 0.2216
    Epoch 3/10
    400/400 [==============================] - 27s 68ms/step - loss: 0.1666 - val_loss: 0.1930
    Epoch 4/10
    400/400 [==============================] - 27s 68ms/step - loss: 0.1504 - val_loss: 0.1822
    Epoch 5/10
    400/400 [==============================] - 28s 70ms/step - loss: 0.1357 - val_loss: 0.2036
    Epoch 6/10
    400/400 [==============================] - 28s 70ms/step - loss: 0.1281 - val_loss: 0.2025
    Epoch 7/10
    400/400 [==============================] - 29s 71ms/step - loss: 0.1226 - val_loss: 0.2172
    


```python

```


```python
# calculate the predictions for the whole validation set

y_pred2=[]
for n in progressbar.progressbar( range(0,x_val_uni.shape[0]) ):
    x=x_val_uni[n,:,:]
    x=x[np.newaxis,:,:]
    y_pred2.append(simple_lstm_model2(x))
    
```

    100% (3803 of 3803) |####################| Elapsed Time: 0:03:31 Time:  0:03:31
    


```python
# reformat for easier visualization
y_pred_n2=np.array([x.numpy()[0][0] for x in y_pred2])
```


```python
# compare y_pred and ground truth

fig=plt.figure(figsize=(16,6))
plt.plot(y_val_uni,label='y_true',alpha=1);
plt.plot(y_pred_n2,label='y_pred (w. present)',color='r',alpha=0.7);
plt.legend(loc=0)
plt.xlabel('time step')
plt.ylabel('C6H6 (norm)');
```

{% raw %}![alt](/assets/ibm_gas_sensor/output_51_0.png){% endraw %}



```python
# compare y_pred and ground truth

fig=plt.figure(figsize=(16,6))
# plt.plot(y_val_uni,label='y_true',alpha=1);
plt.plot(y_pred_n2,label='y_pred (w. present)',color='r',alpha=1);
plt.plot(y_pred_n,'.--',label='y_pred (w/o present)',color='darkgreen',alpha=0.5);
plt.legend(loc=0)
plt.xlabel('time step')
plt.ylabel('C6H6 (norm)');
```

{% raw %}![alt](/assets/ibm_gas_sensor/output_52_0.png){% endraw %}




```python
plt.plot(y_val_uni-y_pred_n,label='error (w/o present)',color='r',alpha=1);
plt.plot(y_val_uni-y_pred_n2,label='error (w. present)',color='b',alpha=.6);

plt.legend(loc=0)
plt.xlabel('time step')
plt.ylabel('error in C6H6 pred. (norm)');
```

{% raw %}![alt](/assets/ibm_gas_sensor/output_53_0.png){% endraw %}



```python
# we compute the same metrics as for the Linear Regression 

print('Test:')
print('MAE: ',mean_absolute_error(y_val_uni, y_pred_n2))
print('MSE: ',mean_squared_error(y_val_uni, y_pred_n2))
print('Max Abs error: ',  np.max(np.abs(y_val_uni- y_pred_n2)))

    Test:
    MAE:  0.21244995687441215
    MSE:  0.10731875691364234
    Max Abs error:  4.084855129497281

```

    

### Model evaluation

The model making use of current sensor data yield even better predictions. 
The goal described in the use case (predicting the concentration of Benzene) is achieved.




