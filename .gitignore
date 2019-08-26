#importing all important libraries

import pandas as pd
import numpy as np
import math
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import numpy as np
from pandas import datetime
from matplotlib import pyplot as plt
import os
from sklearn.preprocessing import MinMaxScaler
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

#load the dataset of given Power-Networks-LCL
df=pd.read_csv("Power-Networks-LCL.csv")   

#now we count house by group by with date time and LCid
housecount = df.groupby('DateTime')[['LCLid']].nunique()
housecount.head(4)
#now plot graph of house count
housecount.plot(figsize=(25,5))

energy = df.groupby('DateTime')[['KWh']].sum()
energy = df.merge(housecount, on = ['DateTime'])
energy = df.reset_index()

energy.count()

energy.DateTime = pd.to_datetime(energy.DateTime,format='%Y-%m-%d').dt.date
energy['avg_energy'] =  energy['KWh']/ len(energy['LCLid'])
print("Starting Point of Data at Day Level",min(energy.DateTime))
print("Ending Point of Data at Day Level",max(energy.DateTime))
energy.describe()

#energy consumption and visibilty graph

fig, ax1 = plt.subplots(figsize = (20,5))
ax1.plot(energy.DateTime,energy.avg_energy, color = 'tab:orange')
ax1.set_ylabel('Avg energy',color = 'tab:orange')
ax2 = ax1.twinx()
ax2.plot(energy.DateTime,energy.avg_energy,color = 'tab:blue')
ax2.set_ylabel('Average Energy/Household',color = 'tab:blue')
plt.title('Energy Consumption and Visibility')
fig.tight_layout()
plt.show()

#training the data model

model_data =energy[['avg_energy','KWh','DateTime']]
train = model_data.iloc[0:(len(model_data)-30)]
test = model_data.iloc[len(train):(len(model_data)-1)]

#now plot the train and test data

train['avg_energy'].plot(figsize=(25,4))
test['avg_energy'].plot(figsize=(25,4))

#now we plot the Autocorelation ghraph

plot_acf(train.avg_energy,lags=100)
plt.show()

#after Autocorelation now we draw the partial Autocorelation 
plot_pacf(train.avg_energy,lags=50)
plt.show()

#model fitting

endog = train['avg_energy']
exog = sm.add_constant(train[['DateTime','KWh']])
n=np.asarray(endog)
m=np.asarray(exog)
mod = sm.tsa.statespace.SARIMAX(endog=n, exog=m, order=(7,1,1),seasonal_order=(1,1, 0, 12),trend='c')
model_fit = mod.fit()
model_fit.summary()
predict = model_fit.predict(start = len(train),end = len(train)+len(test)-1,exog = sm.add_constant(test[['LCLid','stdorToU']]))
test['avg_energy'] = predict.values
test.tail(5)

#now map using MAE and Mape
test['residual'] = abs(test['avg_energy']-test['KWh'])
MAE = test['residual'].sum()/len(test)
MAPE = (abs(test['residual'])/test['avg_energy']).sum()*100/len(test)
print("MAE:", MAE)
print("MAPE:", MAPE)

#now we plot the predicted graph on test value
test['avg_energy'].plot(figsize=(25,10),color = 'red')
test['KWh'].plot()
plt.show()

#now we give train data upto 90% and test 10%

#Subset for required columns and 70-30 train-test split
model_data = energy[['avg_energy','KWh','Acorn']]
train = model_data.iloc[0:round(len(model_data)*0.90)]
test = model_data.iloc[len(train)-1:]
#train = model_data.iloc[0:(len(model_data)-30)]
#test = model_data.iloc[len(train):(len(model_data)-1)]

#ND PLOT THE GRAPH
train['avg_energy'].plot(figsize=(25,4))
test['avg_energy'].plot(figsize=(25,4))

#NOW  ALTERNATE ALGORITHM IS GOINGTO BE USED LSTM AND NORMALISATION 
np.random.seed(11)
dataframe = energy.loc[:,'avg_energy']
dataset = dataframe.values
dataset = dataset.astype('float32')

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg
    
reframed = series_to_supervised(dataset, 7,1)
reframed.head(3)

reframed['KWh'] = energy.KWh.values[7:]
reframed['Acorn_grouped']= energy.Acorn_grouped.values[7:]

reframed = reframed.reindex(['KWh', 'Acorn_grouped','var1(t-7)', 'var1(t-6)', 'var1(t-5)', 'var1(t-4)', 'var1(t-3)','var1(t-2)', 'var1(t-1)', 'var1(t)'], axis=1)
reframed = reframed.values

scaler = MinMaxScaler(feature_range=(0, 1))
reframed = scaler.fit_transform(reframed)

train = reframed[:(len(reframed)-30), :]
test = reframed[(len(reframed)-30):len(reframed), :]

train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]

train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

test_X = test_X.reshape(test_X.shape[0], test_X.shape[2])
nv_yhat = np.concatenate((yhat, test_X), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)

test_y = test_y.reshape((len(test_y), 1))
inv_y = np.concatenate((test_y, test_X), axis=1)
inv_y = scaler.inverse_transform(inv_y)

act = [i[9] for i in inv_y] # last element is the predicted average energy
pred = [i[9] for i in inv_yhat] # last element is the actual average energy

# calculate RMSE
import math
rmse = math.sqrt(mean_squared_error(act, pred))
print('Test RMSE: %.3f' % rmse)

predicted_lstm = pd.DataFrame({'energy':pred,'avg_energy':act})
predicted_lstm['avg_energy'].plot(figsize=(25,10),color = 'red')
predicted_lstm['predicted'].plot(color = 'blue')
plt.show()

