# -*- coding: utf-8 -*-
"""
Created on Thu April 26 09:22:31 2018
@author: Faris Mismar
"""
import keras

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.layers import Dropout

import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler

from pandas.stats import moments
import matplotlib.pyplot as plt

import os
import sys
#os.chdir('C:/Users/ATOC Resource 3/Desktop/Jio/Hackathon')
os.chdir('/Users/farismismar/Dropbox/Stock Trading Using ML')

# Check if tensorflow is used
if (keras.backend.backend() != 'tensorflow' and keras.backend.image_data_format() != 'channels_last' and keras.backend.image_dim_ordering() != 'tf'):
    print('Install tensorflow, configure keras.json to include channels_last for image format and tf for image dimension ordering.')
    print('Program will now exit.')
    sys.exit(1)
    
# Set the random seed
seed = 123
np.random.seed(seed)

# Import the datafile to memory first
TICKER = 'AXISBANK.NS'

dataset = pd.read_csv('./Dataset/{}.csv'.format(TICKER))

# Sanity check. Missing values?
print('Number of missing values: {}'.format(dataset.isnull().sum().sum()))

# Generate more features X for the data
# Ask 1: 
# Simple Moving Average (30, 40, 50 Days) for closing prices
# Reference: https://www.investopedia.com/articles/active-trading/052014/how-use-moving-average-buy-stocks.asp
for i in np.array([30,40,50]):
    sma_i = dataset.rolling(window=i).mean()
    sma_i = dataset[['Close']]
    # Change the column name
    sma_i = sma_i.rename(index=str,columns={'Close': 'Close_SMA_{}'.format(i)})
    # Append it as a new feature
    dataset = dataset.join(sma_i)

# Ask 2: 
# Exponential Moving Average (30, 40, 50 Days) for closing prices
# Reference: https://www.investopedia.com/articles/active-trading/052014/how-use-moving-average-buy-stocks.asp
for i in np.array([30,40,50]):
    ema_i = pd.Series.ewm(dataset['Close'], span=i).mean().to_frame()
    # Change the column name
    ema_i = ema_i.rename(index=str,columns={'Close': 'Close_EMA_{}'.format(i)})
    # Append it as a new feature
    dataset = dataset.join(ema_i)

# Ask 3:
# Aroon Oscillator ( 30, 40, 50 Days) for TBD?
# Reference: TBD
def aroon(df, n=25):
    up = 100 * moments.rolling_apply(df['High'], n + 1, lambda x: x.argmax()) / n
    dn = 100 * moments.rolling_apply(df['Low'], n + 1, lambda x: x.argmin()) / n
    return pd.DataFrame(dict(up=up, down=dn))

for i in np.array([30,40,50]):
    aroon_i = aroon(dataset, i)
    aroon_osc_i = aroon_i['down'] - aroon_i['up']
    aroon_osc_i = aroon_osc_i.to_frame()
    aroon_osc_i.columns = ['aroon_osc_{}'.format(i)]
    # Append it as a new feature
    dataset = dataset.join(aroon_osc_i)

# Ask 4:
# MACD
# Reference: https://github.com/Crypto-toolbox/pandas-technical-indicators/blob/master/technical_indicators.py

def macd(df, n_fast=26, n_slow=12):
    """Calculate MACD, MACD Signal and MACD difference
    
    :param df: pandas.DataFrame
    :param n_fast: 
    :param n_slow: 
    :return: pandas.DataFrame
    """
    EMAfast = pd.Series(df['Close'].ewm(span=n_fast, min_periods=n_slow).mean())
    EMAslow = pd.Series(df['Close'].ewm(span=n_slow, min_periods=n_slow).mean())
    MACD = pd.Series(EMAfast - EMAslow, name='MACD_' + str(n_fast) + '_' + str(n_slow))
    MACDsign = pd.Series(MACD.ewm(span=9, min_periods=9).mean(), name='MACDsign_' + str(n_fast) + '_' + str(n_slow))
    MACDdiff = pd.Series(MACD - MACDsign, name='MACDdiff_' + str(n_fast) + '_' + str(n_slow))
    df = df.join(MACD)
    df = df.join(MACDsign)
    df = df.join(MACDdiff)
    return df

dataset = macd(dataset)

# Ask 5:
# RSI
# Reference: https://github.com/Crypto-toolbox/pandas-technical-indicators/blob/master/technical_indicators.py
# Choice of n = 14 is due to Wilder recommended a smoothing period of 14 (see exponential smoothing, i.e. α = 1/14 or N = 14).
def relative_strength_index(df, n=14):
    """Calculate Relative Strength Index(RSI) for given data.
    
    :param df: pandas.DataFrame
    :param n: 
    :return: pandas.DataFrame
    """
    i = 0
    UpI = [0]
    DoI = [0]
    while i + 1 <= df.index[-1]:
        UpMove = df.loc[i + 1, 'High'] - df.loc[i, 'High']
        DoMove = df.loc[i, 'Low'] - df.loc[i + 1, 'Low']
        if UpMove > DoMove and UpMove > 0:
            UpD = UpMove
        else:
            UpD = 0
        UpI.append(UpD)
        if DoMove > UpMove and DoMove > 0:
            DoD = DoMove
        else:
            DoD = 0
        DoI.append(DoD)
        i = i + 1
    UpI = pd.Series(UpI)
    DoI = pd.Series(DoI)
    PosDI = pd.Series(UpI.ewm(span=n, min_periods=n).mean())
    NegDI = pd.Series(DoI.ewm(span=n, min_periods=n).mean())
    RSI = pd.Series(PosDI / (PosDI + NegDI), name='RSI_' + str(n))
    #df = df.join(RSI)
    return RSI

dataset = dataset.join(relative_strength_index(dataset))

# Ask 6:
# Bollinger Bands ( 30, 40, 50 Days)
# Reference: https://github.com/Crypto-toolbox/pandas-technical-indicators/blob/master/technical_indicators.py

def bollinger_bands(df, n):
    """
    
    :param df: pandas.DataFrame
    :param n: 
    :return: pandas.DataFrame
    """
    MA = pd.Series(df['Close'].rolling(window=n, min_periods=n).mean())
    MSD = pd.Series(df['Close'].rolling(window=n, min_periods=n).std())
    b1 = 4 * MSD / MA
    B1 = pd.Series(b1, name='BollingerB_' + str(n))
    df = df.join(B1)
    b2 = (df['Close'] - MA + 2 * MSD) / (4 * MSD)
    B2 = pd.Series(b2, name='Bollinger%b_' + str(n))
    df = df.join(B2)
    return df

for i in np.array([30,40,50]):
    dataset = bollinger_bands(dataset, i)
	
# Ask 7:
# Stochastic Oscillator (d and k)
def stochastic_oscillator_d(df, n):
    """Calculate stochastic oscillator %D for given data.
    :param df: pandas.DataFrame
    :param n: 
    :return: pandas.DataFrame
    """
    SOk = pd.Series((df['Close'] - df['Low']) / (df['High'] - df['Low']), name='SO%k')
    SOd = pd.Series(SOk.ewm(span=n, min_periods=n).mean(), name='SO%d_' + str(n))
    df = df.join(SOd)
    return df

def stochastic_oscillator_k(df):
    """Calculate stochastic oscillator %K for given data.
    
    :param df: pandas.DataFrame
    :return: pandas.DataFrame
    """
    SOk = pd.Series((df['Close'] - df['Low']) / (df['High'] - df['Low']), name='SO%k')
    df = df.join(SOk)
    return df

dataset = stochastic_oscillator_k(dataset)

for i in np.array([5, 9, 14]):
    dataset = stochastic_oscillator_d(dataset, i)
		
# Ask 8:
# Stochastic momentum Indicator
# https://github.com/kylejusticemagnuson/pyti/blob/master/pyti/stochrsi.py
def stochastic_momentum_ind(df, n):
    """
    StochRSI.
    Formula:
    SRSI = ((RSIt - RSI LOW) / (RSI HIGH - LOW RSI)) * 100
    """
    rsi = relative_strength_index(df, n)[n:]
    stochrsi = [100 * ((rsi[idx] - np.min(rsi[idx+1-n:idx+1])) / (np.max(rsi[idx+1-n:idx+1]) - np.min(rsi[idx+1-n:idx+1]))) for idx in range(n-1, len(rsi))]
    #stochrsi = fill_for_noncomputable_vals(data, stochrsi)
    return stochrsi
'''
    aa = pd.DataFrame(data=np.array([12,13,14,15,12,11,10,9,8,7,10,12,14,16]))
    aa.rolling(window=3).apply(lambda x:x[2]-x[0])
    '''
# Ask 9:
# Chande Momentum Oscillator
# Reference: https://www.investopedia.com/terms/c/chandemomentumoscillator.asp
# https://github.com/kylejusticemagnuson/pyti/tree/master/pyti
def chande_momentum_oscillator(df, n):
    """
    Chande Momentum Oscillator.
    Formula:
    cmo = 100 * ((sum_up - sum_down) / (sum_up + sum_down))
    """

    close_data = np.array(df['Close'])

    moving_period_diffs = [[(close_data[idx+1-n:idx+1][i] -
                 close_data[idx+1-n:idx+1][i-1]) for i in range(1, len(close_data[idx+1-n:idx+1]))] for idx in range(0, len(close_data))]

    sum_up = []
    sum_down = []
    for period_diffs in moving_period_diffs:
        ups = [val if val > 0 else 0 for val in period_diffs]
        sum_up.append(sum(ups))
        downs = [abs(val) if val < 0 else 0 for val in period_diffs]
        sum_down.append(sum(downs))

    sum_up = np.array(sum_up)
    sum_down = np.array(sum_down)

    cmo = pd.Series(100 * ((sum_up - sum_down) / (sum_up + sum_down)), name='Chande_'+str(n))
    df = df.join(cmo)
    return df



# Ask 10:
# Commodity Channel Index (30, 40, 50 Days)

def commodity_channel_index(df, n):
    """Calculate Commodity Channel Index for given data.

    :param df: pandas.DataFrame
    :param n: 
    :return: pandas.DataFrame
    """
    PP = (df['High'] + df['Low'] + df['Close']) / 3
    CCI = pd.Series((PP - PP.rolling(window=n, min_periods=n).mean()) / PP.rolling(window=n, min_periods=n).std(), name='CCI_' + str(n))
    df = df.join(CCI)
    return df
	
for i in np.array([30,40,50]):
    dataset = commodity_channel_index(dataset, i)

# Ask 11:
# Reference: https://www.quantopian.com/posts/technical-analysis-indicators-without-talib-code
# Proof of magnitude is: http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:chaikin_oscillator
def Chaikin_osc(df):  
    ad = (2 * df['Close'] - df['High'] - df['Low']) / (df['High'] - df['Low']) * df['Volume']  
    Chaikin = pd.Series(pd.ewma(ad, span = 3, min_periods = 3) - pd.ewma(ad, span = 10, min_periods = 10), name = 'Chaikin_osc(3,10)')
    df = df.join(Chaikin)

    return df

dataset = Chaikin_osc(dataset)

# Chakin Volatility indicator (30, 40, 50 Days)
# Reference: https://www.linnsoft.com/techind/chaikin-money-flow-cmf
def Chaikin(df,n):
    AD = df['Volume']* (df['Close'] - df['Open']) / (df['High'] - df['Low']) # AD = VOL * (CL - OP) / (HI - LO) #AD stands for Accumulation Distribution
    CMF = pd.Series(AD.rolling(window=n, min_periods=n).sum()) / pd.Series(df['Volume'].rolling(window=n, min_periods=n).sum()) #    CMF = SUM(AD, n) / SUM(VOL, n) where n = Period
    Chaikin = pd.Series(CMF, name='Chaikin_'+str(n))
    Chaikin_dir = pd.Series(Chaikin > 0, name='Chaikin_dir_'+str(n)) + 0
    df = df.join(Chaikin)
    df = df.join(Chaikin_dir)
    
    return df

for i in np.array([30,40,50]):
    dataset = Chaikin(dataset, i)
    
# Ask 12:
# Trend Detection Index (30, 40, 50 Days)
# https://www.linnsoft.com/techind/trend-detection-index-tdi

def trend_detection_index(df, n):     
    '''
    Mom = Price - Price[Period] 
    MomAbs = Abs(Mom) 
    MomSum = Sum(Mom, Period) 
    MomSumAbs = Abs(MomSum) 
    MomAbsSum = Sum(MomAbs, Period) 
    MomAbsSum2 = Sum(MomAbs, Period * 2) 
    TDI = MomSumAbs - (MomAbsSum2 - MomAbsSum)
    '''
    
    Mom = [(df['Close'][idx-n] - df['Close'][idx]) for idx in range(n,len(df['Close']))]
    MomAbs = pd.Series(Mom).abs()
    MomSum = pd.Series(Mom).rolling(window=n, min_periods=n).sum()
    MomSumAbs = MomSum.abs()

    MomAbsSum = pd.Series(MomAbs).rolling(window=n, min_periods=n).sum()
    MomAbsSum2 = pd.Series(MomAbs).rolling(window=2*n, min_periods=2*n).sum()
    
    TDI = pd.Series(MomSumAbs - (MomAbsSum2 - MomAbsSum), name='TDI_'+str(n))
    
    df = df.join(TDI)
    
    return df

for i in np.array([30,40,50]):
    dataset = trend_detection_index(dataset, i)
    
# Ask 13:
# Rate of Price Change (30, 40, 50 Days)
def rate_of_price_change(df, n):
    """
    
    :param df: pandas.DataFrame
    :param n: 
    :return: pandas.DataFrame
    """
    M = df['Close'].diff(n - 1)
    N = df['Close'].shift(n - 1)
    ROC = pd.Series(M / N, name='ROPC_' + str(n))
    df = df.join(ROC)
    return df
	
for i in np.array([30,40,50]):
    dataset = rate_of_price_change(dataset, i)

# Ask 14:
# Rate of Volume Change (30, 40, 50 Days)
def rate_of_volume_change(df, n):
    """
    
    :param df: pandas.DataFrame
    :param n: 
    :return: pandas.DataFrame
    """
    M = df['Volume'].diff(n - 1)
    N = df['Volume'].shift(n - 1)
    ROC = pd.Series(M / (0.00001 + N), name='ROVC_' + str(n))
    df = df.join(ROC)
    return df
	
for i in np.array([30,40,50]):
    dataset = rate_of_volume_change(dataset, i)

# Ask 15:
# William %R (30, 40, 50 Days)
# Code written after: https://tradingsim.com/blog/williams-percent-r/
# and http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:williams_r
def william_r(df, n):
    """Calculate William %R for given data.
    
    :param df: pandas.DataFrame
    :return: pandas.DataFrame
    """
    high=df['High'].rolling(window=n).max()
    low=df['Low'].rolling(window=n).min()
    
    william = pd.Series((high - df['Close']) / (high - low), name='William%R_' + str(n)) * -100.
    df = df.join(william)
    return df

for i in np.array([30,40,50]):
    dataset = william_r(dataset, i)

############################################################################################
# Save the file
dataset.to_csv(path_or_buf='./Dataset/{}_complete.csv'.format(TICKER), index=False)

# Plot the data closing
plt.figure(figsize=(15,3))
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.plot(np.log10(dataset['Close'].values))
plt.title('$\ln$ Closing Price')
plt.grid()
plt.show()

############################################################################################

# Keep a copy of the original file in case
dataframe = dataset
dataset = dataset.drop(['Date'], axis=1)

# Perform a split 70-30
m, n = dataset.shape
rsplit = 0.7

# split into train and test sets
train_size = int(rsplit * m)
test_size = m - train_size

# Introduce ln to the closing price to make it look stationary
#dataset['Close'] = np.log(dataset['Close'])

# Replace all NaNs with -12345
dataset.fillna(value=-12345,inplace=True)

train, test = dataset.iloc[0:train_size,:], dataset.iloc[train_size:m,:]

X_train = train.drop(['Close'], axis = 1)
X_test = test.drop(['Close'], axis = 1)
y_train = train['Close']
y_test = test['Close']

# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

X_train_sc = X_train_sc.astype('float32')
############################################################################################

# Generate LSTM
# Check this:
# https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/ (BS)
# https://towardsdatascience.com/time-series-analysis-in-python-an-introduction-70d5a5b1d52a
# https://towardsdatascience.com/using-lstms-to-forecast-time-series-4ab688386b1f

# Generate lag data
# https://machinelearningmastery.com/tune-lstm-hyperparameters-keras-time-series-forecasting/

def difference(df, lag=1):
	columns = [df.shift(i) for i in np.arange(1, lag+1)]
	columns.append(df)
	df = pd.concat(columns, axis=1)
	return df

def rmse(y_true, y_pred):
	return keras.backend.sqrt(keras.backend.mean(keras.backend.square(y_pred - y_true), axis=-1))
 

mX, nX = X_train.shape
mY, = y_train.shape
time_steps = 1
batch_size = 2

# reshape input to be [samples, time steps, features]
X_train_sc = np.reshape(X_train_sc, (mX, time_steps, nX))
X_test_sc = np.reshape(X_test_sc, (X_test_sc.shape[0], time_steps, nX))

# reshape output too to be one feature.
y_train = np.reshape(y_train, (mY, 1))
y_test = np.reshape(y_test, (-1, 1))

# Build LSTM
'''
model = Sequential()
model.add(LSTM(input_shape=(time_steps, nX), units=nX, activation='relu', return_sequences=True, stateful=True, batch_input_shape=(batch_size, time_steps, nX)))
#model.add(LSTM(batch_size, return_sequences=True, stateful=True))
model.add(LSTM(batch_size, stateful=True))
#model.add(Dropout(0.2))
#model.add(Dense(10, activation='relu'))
model.add(Dense(1)) # no matter what, do not change this.  This is since y is a vector. 
model.add(Activation('linear'))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse', 'mae', 'mape'])
'''
model = Sequential()
model.add(LSTM(input_shape=(time_steps, nX), units=nX, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1)) # no matter what, do not change this.  This is since y is a vector. 
model.add(Activation('linear'))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=[rmse])
model.summary()

model.fit(X_train_sc, y_train, epochs=256, batch_size=batch_size, verbose=2)

# Generate a timeseries from 2018-04-04 to 2018-04-04 + 30 days = 2018-05-04
# Use data from Yahoo finance to predict these stocks 
y_hat = model.predict(X_test_sc, batch_size=batch_size)

plt.figure(figsize=(15,3))
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plot_test, = plt.plot(y_test, label='Test data')
plot_predicted, = plt.plot(y_hat, label='Predicted data')
plt.legend([plot_test, plot_predicted])
plt.grid(True)
plt.xlim([0,y_hat.shape[0]])
plt.ylabel('Price in USD')
plt.xlabel('Date')
plt.title('Predicted Closing Price')
plt.show()

############################################################################################    
# This is the function that generates the requirement
def generate_return(closing_0, closing_30):
    return (closing_30 - closing_0) / closing_0 * 100.
    
############################################################################################
'''
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.preprocessing import StandardScaler

ss = StandardScaler()

X_train_sc = ss.fit_transform(X_train)
X_test_sc = ss.transform(X_test)

# create model
def create_mlp(intermediate_dim, n_hidden):
    mlp = Sequential()
    mlp.add(Dense(units=intermediate_dim, input_dim=nX, activation='relu'))
    for k in np.arange(n_hidden):
        mlp.add(Dense(intermediate_dim, use_bias=False))

    mlp.add(Dense(units=nY, input_dim=intermediate_dim, activation='relu', use_bias=True))

    mlp.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy'])
    return mlp

model = KerasRegressor(build_fn=create_mlp, verbose=1, epochs=32, batch_size=32)

# The hyperparameters
dims = [100,1000]#,5000,10000]
n_hiddens = [50, 500] #,500,1000]

hyperparameters = dict(intermediate_dim=dims, n_hidden=n_hiddens)

grid = GridSearchCV(estimator=model, param_grid=hyperparameters, n_jobs=1, cv=3)
grid_result = grid.fit(X_train_sc, Y_train)

# This is the best model
best_model_mlp = grid_result.best_params_

mlp = Sequential()
mlp.add(Dense(units=grid_result.best_params_['intermediate_dim'], input_dim=nX, activation='relu'))

for k in np.arange(grid_result.best_params_['n_hidden']):
    mlp.add(Dense(grid_result.best_params_['intermediate_dim'], use_bias=False)) # no sigmoid here

mlp.add(Dense(units=nY, input_dim=grid_result.best_params_['intermediate_dim'], activation='relu', use_bias=True))

# Compile model with accuracy metric
mlp.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy'])
mlp.fit(X_train_sc, Y_train, epochs=32, batch_size=32)

# Create mlp object with params from grid_result then generate these
y_hat = mlp.predict(X_test_sc)

# Now score the model for accuracy
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score

Y_test = Y_test.values
y1_r2 = r2_score(Y_test[:,0], y_hat[:,0])
y2_r2 = r2_score(Y_test[:,1], y_hat[:,1])

y1_acc = accuracy_score(Y_test[:,0], y_hat[:,0])
y2_acc = accuracy_score(Y_test[:,1], y_hat[:,1])

print(best_model_mlp)
print('R-sq for y1 dataset is: {0:.3f} and for y2 dataset is {1:.3f}.'.format(y1_r2, y2_r2))
print('Accuracy for y1 dataset is: {0:.4f} and for for y2 dataset is {1:.4f}'.format(y1_acc, y2_acc))
'''
