#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 20:48:10 2022

@author: farismismar
"""
import os

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.compat.v1 import set_random_seed

from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import RandomOverSampler

import tensorflow as tf

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import datetime
from io import StringIO
import requests

import re

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

if os.name == 'nt':
    os.add_dll_directory("/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.5/bin")

import tensorflow as tf
#print(tf.config.list_physical_devices('GPU'))

# The GPU ID to use, usually either "0" or "1" based on previous line.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"   # My NVIDIA GeForce RTX 3050 Ti GPU output from line 31

import pdb

class StockPricePredictor:
    ver = '0.4'
    rel_date = '2022-09-11'
    
    # OK
    def __init__(self, ticker, seed):
        self.ticker = ticker
                
        # Fix the seed for reproducibility
        self.seed = seed
        self.reset_seed()


    def __ver__(self):
        return self.ver, self.rel_date
        

    def reset_seed(self):
        seed = self.seed
        self.np_random = np.random.RandomState(seed=seed)
        set_random_seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        return self
    
    
    # OK
    def load_data(self):
        # Fetch the data online
                
        TICKER = self.ticker
        start_date = 'March 1, 2000'
        
        start_date = datetime.datetime.strptime(start_date, '%B %d, %Y')
        end_date = datetime.datetime.today() - datetime.timedelta(days=1) # yesterday's date
        
        
        # Now fetch the stock price
        start_unix = start_date.strftime('%s')
        end_unix = end_date.strftime('%s')
        
        fetch_url = f'https://query1.finance.yahoo.com/v7/finance/download/{TICKER}?period1={start_unix}&period2={end_unix}&interval=1d&events=history&includeAdjustedClose=true'
        
        s = requests.get(fetch_url).text
        
        if (s == 'Forbidden'):
            print(f'ERROR: {s} accessing {fetch_url}')
            return pd.DataFrame()
            
        df_ticker = pd.read_csv(StringIO(s))
        
        pdb.set_trace()
        
        df_ticker['Date'] = pd.to_datetime(df_ticker['Date'], format='%Y-%m-%d').dt.strftime('%Y-%m-%d')
        del s
        
        # Now fetch the treasury yield
        fetch_url = 'https://www.treasury.gov/resource-center/data-chart-center/interest-rates/pages/TextView.aspx?data=yieldAll'
        
        response = requests.get(fetch_url)
        data = response.text
        
        date_pattern = re.compile('[0-9]{2}/[0-9]{2}/[0-9]{2}', re.MULTILINE)
        
        dates = pd.DataFrame(data={'Dates': re.findall(date_pattern, data)})
        dates['Dates'] = pd.to_datetime(dates['Dates'], format='%m/%d/%y').dt.strftime('%Y-%m-%d')
        dates = dates.iloc[1:,:].reset_index(drop=True)
        
        values_pattern = re.compile('(>[0-9]{,3}\.[0-9]{2}<|N/A)')
        numbers = pd.DataFrame(data={'Values': re.findall(values_pattern, data)})
        
        headers_pattern = re.compile('>([A-Za-z\s0-9]+)</th>')
        headers = re.findall(headers_pattern, data)
        del data
        
        numbers['Values'] = numbers['Values'].apply(lambda x: re.sub('>|<', '', x))
        numbers['Values'] = pd.to_numeric(numbers['Values'], errors='coerce')
          
        numbers = numbers.values
        numbers = pd.DataFrame(np.reshape(numbers, (-1, len(headers) - 1)))
        
        assert(numbers.shape[0] == dates.shape[0])
        
        df_treasury = pd.concat([dates, numbers], axis=1, ignore_index=True)
        df_treasury.columns = headers
        
        # TODO: Now fetch the inflation rates, unemployment rates, DOW, NASDAQ, and S&P 500
        
        # Finally, merge all tables
        df = pd.merge(df_ticker, df_treasury, how='inner', on='Date')
        df = df.set_index('Date')
        
        df.to_csv(f'{TICKER}.csv', sep=',', index=True)
        
        # Date as a datetime object.
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Sanity check. Missing values?
        print('Number of missing values: {}.  Assume previous day data'.format(df.isnull().sum().sum()))

        df.fillna(method='ffill', inplace=True)
        
        return df


    def plot_data(self, df, ticker):
        # Plot the data closing
        fig = plt.figure(figsize=(15,3))
        plt.plot(df['Date'], df['Close'])
        ax = fig.gca()
        ax.xaxis.set_major_locator(plt.MaxNLocator(15)) # Too many dates.
        plt.title('Closing Price')
        plt.xlabel('Date')
        plt.ylabel('Price [USD]')
        plt.grid()
        plt.tight_layout()
        plt.savefig('price_{}.pdf'.format(ticker), format='pdf', dpi=fig.dpi)
        #plt.show()
        plt.close(fig)
        
        
    def plot_learning_history(self, history, title):
        # Plot the losses vs epoch here        
        fig = plt.figure(figsize=(8, 5))
        
        plt.rcParams['font.family'] = "Arial"
        plt.rcParams['font.size'] = "14"
        
        plot1, = plt.plot(history.epoch, history.history['loss'], c='blue')
        plot2, = plt.plot(history.epoch, history.history['val_loss'], linestyle='--', c='blue')
        plt.grid(which='both', linestyle='--')
        
        ax = fig.gca()    
        ax_sec = ax.twinx()
        plot3, = ax_sec.plot(history.epoch, history.history['mse'], lw=2, c='red')
        plot4, = ax_sec.plot(history.epoch, history.history['val_mse'], linestyle='--', lw=2, c='red')
        
        ax.set_xlabel(r'Epoch')
        ax.set_ylabel(r'Loss')
        ax_sec.set_ylabel(r'MSE')
        plt.legend([plot1, plot2, plot3, plot4], [r'Training Loss', r'Validation Loss', r'Training MSE', r'Validation MSE'],
                    bbox_to_anchor=(-0.1, -0.02, 1.20, 1), bbox_transform=fig.transFigure, 
                    loc='lower center', ncol=4, mode="expand", borderaxespad=0.)
        
        plt.title(title)
        plt.tight_layout()
        #tikzplotlib.save(f'output/{title}.tikz')
        plt.savefig(f'{title}.pdf', format='pdf', dpi=fig.dpi)
        #plt.show()
        plt.close(fig)
        


    def plot_prices(self, X_test, y_test, y_pred_nn, lookahead):
        fig = plt.figure(figsize=(15,4))
        plot_actual, = plt.plot(X_test.index, X_test['Close_t'], linewidth=2.75, label='True test data')
        plot_test, = plt.plot(y_test, label=f'True lookforward-{lookahead} test data') 
        plot_predicted, = plt.plot(y_test.index, y_pred_nn, label=f'Predicted lookforward-{lookahead} data') 
        plt.legend()
        ax = fig.gca()
        ax.xaxis.set_major_locator(plt.MaxNLocator(10)) # Too many dates.
        plt.grid(True)
        plt.title('Closing Price')
        plt.xlabel('Date')
        plt.ylabel('Price [USD]')
        fig.tight_layout()
        plt.savefig('prediction_{}.pdf'.format(ticker), format='pdf')
        #plt.show()
        plt.close(fig)


    # https://medium.com/codex/algorithmic-trading-with-stochastic-oscillator-in-python-7e2bec49b60d
    def create_trade_strategy(self, df):
        # TODO
        return
    
    
    def run_prediction(self, target_variable, train_size, lookahead, lookbacks, ticker,
                       epoch_count, batch_size):
        
        df = self.load_data()
        
        if df.shape[0] == 0:
            return None
        
        df_eng = self.feature_engineering(df)
        
        self.plot_data(df, ticker)
        
        # We are predicting 30 days using lookback
        df_eng, engineered_label = self.timeseries_engineering(df_eng, target_variable=target_variable, lookahead=lookahead, lookbacks=lookbacks, dropna=False)
        X_train, X_test, y_train, y_test, _ = self.train_test_split_time(df_eng, engineered_label, time_steps=1, train_size=train_size)
        
        history, model, y_pred_nn = self.train_nn(X_train, X_test, y_train, y_test, lookbacks, epoch_count=epoch_count, 
                                 batch_size=batch_size, callback=None, verbose=True)
        
        self.plot_learning_history(history, title=f'For Stock + {lookahead} using {lookbacks} lookbacks')
        
        # Objective: green line to be 100% on the blue line.
        self.plot_prices(X_test, y_test, y_pred_nn, lookahead)
        
        ############################################################################################
        # Now compute the return for the next 30 days        
        y_last_date = y_pred_nn[-1][0] 
        y_first_date = df['Close'].iloc[-1] # this is 4/4/2018
        
        generated_return = self.generate_return(y_first_date, y_last_date)
        
        # Conclude
        file = open('return_{}.txt'.format(ticker),'w') 
        print('For ticker {0}, the forecasted return is {1:3f}%'.format(ticker, generated_return))
        file.close()
        
        ############################################################################################
        # Save the file
        df_eng.to_csv(path_or_buf='./Dataset/{}_complete.csv'.format(ticker), index=False)


    # OK
    def feature_engineering(self, df):
        # Generate more features X for the data
        df_sma = self.sma(df)
        df_ema = self.ema(df)
        df_aroon = self.aroon(df, n_days=[30,40,50])
        df_macd = self.macd(df, n_fast=26, n_slow=12)
        
        df_rsi = self.relative_strength_index(df, n=14)
        df_bollinger = self.bollinger_bands(df, days=[30,40,50])
        df_SO_d = self.stochastic_oscillator_d(df, n=[5,9,14])
        df_SO_k = self.stochastic_oscillator_k(df)
    
        df_st_m_ind = self.stochastic_momentum_ind(df, period=14)
        
        df_ch_m_o = self.chande_momentum_oscillator(df, period=14)
        
        df_cci = self.commodity_channel_index(df, n_days=[30,40,50])
        df_Chaikin = self.Chaikin(df, n_days=[30,40,50])
        df_Chaikin_osc = self.Chaikin_osc(df)

        # and this function
        df_tdi = self.trend_detection_index(df, n_days=[30,40,50])
        
        df_rpc = self.rate_of_price_change(df, n_days=[30,40,50])
        df_rvc = self.rate_of_volume_change(df, n_days=[30,40,50])
    
        df_williams = self.william_r(df, n_days=[30,40,50])
        
        # Add up the features
        df_features = df.copy()
        for df_ in [df_sma, df_ema, df_aroon, df_macd, df_rsi, df_bollinger, df_SO_d, df_SO_k, df_st_m_ind, df_ch_m_o,
                    df_cci, df_Chaikin, df_Chaikin_osc, df_tdi, df_rpc, df_rvc, df_williams]:
            df_features = pd.concat([df_features, df_], axis=1)
        
        # Finally, set up the time series by using Date as the time index.
        df_features = df_features.set_index('Date')
        
        return df_features
    
    
    def timeseries_engineering(self, df, target_variable, lookahead, lookbacks, dropna=False):
        df_ = df.copy()
        df_y = df_[target_variable].to_frame()
        
        # This is needed in case the idea is to predict y(t), otherwise
        # the data will contain y_t in the data and y_t+0, which are the same
        # and predictions will be trivial.
        if lookahead == 0:
            df_ = df_.drop(target_variable, axis=1)
        
        df_postamble = df_.add_suffix('_t')
        df_postamble = pd.concat([df_postamble, pd.DataFrame(np.zeros_like(df_), index=df_.index, columns=df_.columns).add_suffix('_d')], axis=1)
        
        df_shifted = pd.DataFrame()
        # Noting that column order is important
        for i in range(lookbacks, 0, -1):
            df_shifted_i = df_.shift(i).add_suffix('_t-{}'.format(i))
            df_diff_i = df_.diff(i).add_suffix('_d-{}'.format(i)) # difference with previous time
            
            df_shifted = pd.concat([df_shifted, df_shifted_i, df_diff_i], axis=1)
            
        df_y_shifted = df_y.shift(-lookahead).add_suffix('_t+{}'.format(lookahead))
        df_output = pd.concat([df_shifted, df_postamble, df_y_shifted], axis=1)
        
        if dropna:
            df_output.dropna(inplace=True)
        else:
            # Do not drop data in a time series.  Instead, fill last value
            df_output.fillna(method='bfill', inplace=True)
           
            # Then fill the first value!
            df_output.fillna(method='ffill', inplace=True)
           
            # Drop whatever is left.
            df_output.dropna(how='any', axis=1, inplace=True)
       
        # Whatever it is, no more nulls shall pass!
        assert(df_output.isnull().sum().sum() == 0)
    
        engineered_target_variable = f'{target_variable}_t+{lookahead}'
        
        return df_output, engineered_target_variable
    

    def train_test_split_time(self, df, label, time_steps, train_size, rebalance=False):
        # Avoid truncating training data or test data...
        y = df[label] # must be categorical
        X = df.drop(label, axis=1)
        
        # Balance through ROS (SMOTE did not work due to n_neigh > n_samples)
        if rebalance:
            try:
                print('Oversampling to balance classes...')
                ros = RandomOverSampler(random_state=self.seed)
                X, y = ros.fit_resample(X, y)
            except Exception as e:
                print(f'WARNING: Oversampling failed due to {e}.  No rebalancing performed.')
                raise ValueError()
        else:
            print('No class rebalance will be made (this is time series data).')
            
        # Split on boundary of time
        m = int(X.shape[0] / time_steps * train_size)
        train_rows = int(m * time_steps)
        
        test_offset = ((X.shape[0] - train_rows) // time_steps) * time_steps
        X_train = X.iloc[:train_rows, :]
        X_test = X.iloc[train_rows:(train_rows+test_offset), :]
        y_train = y[:train_rows]
        y_test = y[train_rows:(train_rows+test_offset)]

        return X_train, X_test, y_train, y_test, X_test.index
    

    def train_nn(self, X_train, X_test, y_train, y_test, lookbacks, epoch_count, batch_size, callback=None, verbose=True):
        # Store number of learning features
        mX, nX = X_train.shape
        
        # Scale X features
        sc = MinMaxScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)

        # Now, reshape input to be 3-D: [timesteps, batch, feature]
        X_train = np.reshape(X_train, (-1, lookbacks + 1, X_train.shape[1] // (lookbacks + 1)))
        X_test = np.reshape(X_test, (-1, lookbacks + 1, X_test.shape[1] // (lookbacks + 1)))
        
        # # Now, reshape input to be 3-D: [batch, timesteps, feature]
        # X_train = np.reshape(X_train, (lookbacks + 1, -1, X_train.shape[1] // (lookbacks + 1)))
        # X_test = np.reshape(X_test, (lookbacks + 1, -1, X_test.shape[1] // (lookbacks + 1)))
    
        print('INFORMATION: Starting optimization...')
        
        model = self._create_lstm_nn(input_shape=(X_train.shape[1], X_train.shape[2]),
                                      output_shape=(X_train.shape[1], 1))
        
        #patience = 2 * lookbacks
        callback_list = [] # EarlyStopping(monitor='val_loss', min_delta=0, patience=patience)]
        
        if callback is not None:
            callback_list = callback_list + [callback]
            
        history = model.fit(X_train, y_train, epochs=epoch_count, batch_size=batch_size, 
                            callbacks=callback_list,
                            validation_data=(X_test, y_test), 
                            #validation_split=0.5, # cannot do this for time series due to randomness
                            verbose=verbose, shuffle=True)
        
        y_pred_test = model.predict(X_test, batch_size=batch_size) 
            
        return history, model, y_pred_test
    
    def _rmse(self, y_true, y_pred):
        return keras.backend.sqrt(keras.backend.mean(keras.backend.square(y_pred - y_true), axis=-1))


    def _create_lstm_nn(self, input_shape, output_shape, depth=8, width=16):
        mX, nX = input_shape
        
        model = keras.Sequential()
        model.add(layers.LSTM(input_shape=(mX, nX), recurrent_dropout=0.8,
                              units=width, return_sequences=False, 
                              activation="relu",
                              recurrent_activation="sigmoid"))

        for hidden in np.arange(depth):
            model.add(layers.Dense(width, activation='relu'))
        
        model.add(layers.Dropout(0.2))
        model.add(layers.Dense(1, activation='linear'))
        
        
        # This is a good model
        # model.add(layers.LSTM(input_shape=(mX, nX), units=width, activation='relu'))
        # model.add(layers.Dropout(0.2))
        # model.add(layers.Dense(1)) # no matter what, do not change this.  This is since y is a vector. 
        
        model.compile(loss=self._rmse, optimizer='adam', metrics=['mse'])
        
        # Reporting the number of parameters
        print(model.summary())
    
        num_params = model.count_params()
        print('Number of parameters: {}'.format(num_params))
        
        return model


    
    # Ask 1: 
    # Simple Moving Average (30, 40, 50 Days) for closing prices
    # Reference: https://www.investopedia.com/articles/active-trading/052014/how-use-moving-average-buy-stocks.asp
    # OK
    def sma(self, df, days=[30,40,50]):
        df_output = pd.DataFrame()
        for i in np.array(days):
            sma_i = df.rolling(window=i).mean()
            sma_i = sma_i[['Close']]
            # Change the column name
            sma_i = sma_i.rename(columns={'Close': 'Close_SMA_{}'.format(i)})
            df_output = pd.concat([df_output, sma_i], axis=1)
            
        return df_output
    
    
    # Ask 2: 
    # Exponential Moving Average (30, 40, 50 Days) for closing prices
    # Reference: https://www.investopedia.com/articles/active-trading/052014/how-use-moving-average-buy-stocks.asp
    # OK
    def ema(self, df, days=[30,40,50]):
        df_output = pd.DataFrame()
        for i in np.array(days):
            ema_i = pd.Series.ewm(df['Close'], span=i).mean().to_frame()
            # Change the column name
            ema_i = ema_i.rename(columns={'Close': 'Close_EMA_{}'.format(i)})
            # Append it as a new feature
            df_output = pd.concat([df_output, ema_i], axis=1)
        
        return df_output
        
    
    # Ask 3:
    # Aroon Oscillator ( 30, 40, 50 Days) for TBD?
    # Reference: TBD
    # OK
    def aroon(self, df, n_days):
        df_output = pd.DataFrame()
        for n in n_days:
            # up = 100 * pd.rolling_apply(df['High'], n + 1, lambda x: x.argmax()) / n
            # dn = 100 * pd.rolling_apply(df['Low'], n + 1, lambda x: x.argmin()) / n
            # Deprecated are replaced
            up = 100 * df['High'].rolling(n + 1).apply(lambda x: x.argmax()) / n
            dn = 100 * df['Low'].rolling(n + 1).apply(lambda x: x.argmin()) / n
            
            aroon_osc_i = dn - up
            aroon_osc_i = aroon_osc_i.to_frame()
            aroon_osc_i.columns = ['aroon_osc_{}'.format(n)]
            df_output = pd.concat([df_output, aroon_osc_i], axis=1)
            
        return df_output

    
    # Ask 4:
    # MACD
    # Reference: https://github.com/Crypto-toolbox/pandas-technical-indicators/blob/master/technical_indicators.py
    # OK
    def macd(self, df, n_fast=26, n_slow=12):
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
        df_output = MACD.to_frame()
        df_output = pd.concat([df_output, MACDsign, MACDdiff], axis=1)
        
        return df_output


    # Ask 5:
    # RSI
    # Reference: https://github.com/Crypto-toolbox/pandas-technical-indicators/blob/master/technical_indicators.py
    # Choice of n = 14 is due to Wilder recommended a smoothing period of 14 (see exponential smoothing, i.e. α = 1/14 or N = 14).
    # OK
    def relative_strength_index(self, df, n=14):
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
        df_output = pd.Series(PosDI / (PosDI + NegDI), name='RSI_' + str(n)).to_frame()
                
        return df_output
    
    
    # Ask 6:
    # Bollinger Bands ( 30, 40, 50 Days)
    # Reference: https://github.com/Crypto-toolbox/pandas-technical-indicators/blob/master/technical_indicators.py
    # OK
    def bollinger_bands(self, df, days=[30,40,50]):
        """
        
        :param df: pandas.DataFrame
        :param n: 
        :return: pandas.DataFrame
        """
        df_output = pd.DataFrame()
        for n in days:
            MA = pd.Series(df['Close'].rolling(window=n, min_periods=n).mean())
            MSD = pd.Series(df['Close'].rolling(window=n, min_periods=n).std())
            b1 = 4 * MSD / MA
            B1 = pd.Series(b1, name='BollingerB_' + str(n))
            df = df.join(B1)
            b2 = (df['Close'] - MA + 2 * MSD) / (4 * MSD)
            B2 = pd.Series(b2, name='Bollinger%b_' + str(n))
            df_output = pd.concat([df_output, B2], axis=1)
            
        return df_output
    
    	
    # Ask 7:
    # Stochastic Oscillator (d and k)
    def stochastic_oscillator_d(self, df, n=[5,9,14]):
        SOk = pd.Series((df['Close'] - df['Low']) / (df['High'] - df['Low']), name='SO%k')

        df_output = pd.DataFrame()
        for i in n:
            SOd = pd.Series(SOk.ewm(span=i, min_periods=i).mean(), name='SO%d_' + str(i))
            df_output = pd.concat([df_output, SOd], axis=1)    
        
        return df_output
    

    # OK
    def stochastic_oscillator_k(self, df):
        SOk = pd.DataFrame((df['Close'] - df['Low']) / (df['High'] - df['Low']), columns=['SO%k'])
        
        return SOk
    
    		
    # Ask 8:
    # Stochastic momentum Indicator
    # Modified after this:
    # https://github.com/kylejusticemagnuson/pyti/blob/master/pyti/stochrsi.py
    # OK
    def stochastic_momentum_ind(self, df, period):
        """
        StochRSI.
        Formula:
        SRSI = ((RSIt - RSI LOW) / (RSI HIGH - LOW RSI)) * 100
        """
        
        rsi = self.relative_strength_index(df, period)
         
        #stochrsi_old = [(idx+1-period, idx+1) for idx in range(period-1, len(rsi))]
 
        rolling_minima = rsi.rolling(period + 1).min() 
        rolling_maxima = rsi.rolling(period + 1).max()
        
        stochrsi = 100 * (rsi - rolling_minima) / (rolling_maxima - rolling_minima)

        return stochrsi


    # Ask 9:
    # Chande Momentum Oscillator
    # Reference: https://www.investopedia.com/terms/c/chandemomentumoscillator.asp
    # https://github.com/kylejusticemagnuson/pyti/tree/master/pyti
    # Not OK.  Needs optimization
    def chande_momentum_oscillator(self, df, period):
        """
        Chande Momentum Oscillator.
        Formula:
        cmo = 100 * ((sum_up - sum_down) / (sum_up + sum_down))
        """
    
        close_data = np.array(df['Close'])
    
        moving_period_diffs = [[(close_data[idx+1-period:idx+1][i] -
                     close_data[idx+1-period:idx+1][i-1]) for i in range(1, len(close_data[idx+1-period:idx+1]))] for idx in range(0, len(close_data))]
    
        sum_up = []
        sum_down = []
        for period_diffs in moving_period_diffs:
            ups = [val if val > 0 else 0 for val in period_diffs]
            sum_up.append(sum(ups))
            downs = [abs(val) if val < 0 else 0 for val in period_diffs]
            sum_down.append(sum(downs))
    
        sum_up = np.array(sum_up)
        sum_down = np.array(sum_down)
    
        cmo = pd.Series(100 * ((sum_up - sum_down) / (sum_up + sum_down)), name='Chande_'+str(period))
    
        return cmo
    
    
    # Ask 10:
    # Commodity Channel Index (30, 40, 50 Days)
    # OK
    def commodity_channel_index(self, df, n_days=[30,40,50]):
        """Calculate Commodity Channel Index for given data.
        :param df: pandas.DataFrame
        :param n: 
        :return: pandas.DataFrame
        """
        df_output = pd.DataFrame()
        for n in n_days:
            PP = (df['High'] + df['Low'] + df['Close']) / 3
            CCI = pd.Series((PP - PP.rolling(window=n, min_periods=n).mean()) / PP.rolling(window=n, min_periods=n).std(), name='CCI_' + str(n))
            df_output = pd.concat([df_output, CCI], axis=1)
            
        return df_output
    	
    
    # Ask 11:
    # Proof of method is: http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:chaikin_oscillator
    # OK
    def Chaikin_osc(self, df, periods=14):  
        money_flow_mult = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'])
        money_flow_vol = money_flow_mult * df['Volume']
        
        mfv3 = money_flow_vol.ewm(min_periods=3, adjust=True, span=3, ignore_na=False).mean()
        mfv10 = money_flow_vol.ewm(min_periods=10, adjust=True, span=10, ignore_na=False).mean()
        
        Chaikin = pd.DataFrame(mfv3 - mfv10, columns=['Chaikin_osc(3,10)'])
            
        return Chaikin

    
    # Chakin Volatility indicator (30, 40, 50 Days)
    # Reference: https://www.linnsoft.com/techind/chaikin-money-flow-cmf
    # OK
    def Chaikin(self, df, n_days):
        df_output = pd.DataFrame()
        AD = df['Volume'] * (df['Close'] - df['Open']) / (df['High'] - df['Low']) # AD = VOL * (CL - OP) / (HI - LO) #AD stands for Accumulation Distribution
        for n in n_days:        
            CMF = pd.Series(AD.rolling(window=n, min_periods=n).sum()) / pd.Series(df['Volume'].rolling(window=n, min_periods=n).sum()) #    CMF = SUM(AD, n) / SUM(VOL, n) where n = Period
            Chaikin = pd.Series(CMF, name='Chaikin_'+str(n))
            Chaikin_dir = pd.Series(Chaikin > 0, name='Chaikin_dir_'+str(n)) + 0
            df_output = pd.concat([df_output, Chaikin, Chaikin_dir], axis=1)
            
        return df_output
    
        
    # Ask 12:
    # Trend Detection Index (30, 40, 50 Days)
    # https://www.linnsoft.com/techind/trend-detection-index-tdi
    # OK.
    def trend_detection_index(self, df, n_days):
        df_output = pd.DataFrame()
        for n in n_days:
            #Mom = [(df['Close'][idx-n] - df['Close'][idx]) for idx in range(n,len(df['Close']))]
            Mom = df['Close'].shift(n) - df['Close']
            MomAbs = Mom.abs()
            MomSum = Mom.rolling(window=n, min_periods=n).sum()
            #MomSumAbs = MomSum.abs() # unneeded step
        
            MomAbsSum = pd.Series(MomAbs).rolling(window=n, min_periods=n).sum()
            MomAbsSum2 = pd.Series(MomAbs).rolling(window=2*n, min_periods=2*n).sum()
            
            TDI = pd.Series(MomSum - (MomAbsSum2 - MomAbsSum), name='TDI_'+str(n))
        
            df_output = pd.concat([df_output, TDI], axis=1)
        
        return df_output
    
    
    # Ask 13:
    # Rate of Price Change (30, 40, 50 Days)
    # OK
    def rate_of_price_change(self, df, n_days):
        """
        :param df: pandas.DataFrame
        :param n: 
        :return: pandas.DataFrame
        """
        df_output = pd.DataFrame()
        for n in n_days:
            M = df['Close'].diff(n - 1)
            N = df['Close'].shift(n - 1)
            ROC = pd.Series(M / N, name='ROPC_' + str(n))
            df_output = pd.concat([df_output, ROC], axis=1)
        
        return df_output
    	
    
    # Ask 14:
    # Rate of Volume Change (30, 40, 50 Days)
    # OK
    def rate_of_volume_change(self, df, n_days):
        """
        
        :param df: pandas.DataFrame
        :param n: 
        :return: pandas.DataFrame
        """
        eps = 1e-5
        df_output = pd.DataFrame()
        for n in n_days:
            M = df['Volume'].diff(n - 1)
            N = df['Volume'].shift(n - 1)
            ROC = pd.Series(M / (eps + N), name='ROVC_' + str(n))
            df_output = pd.concat([df_output, ROC], axis=1)
            
        return df_output
    	
    
    # Ask 15:
    # William %R (30, 40, 50 Days)
    # Code written after: https://tradingsim.com/blog/williams-percent-r/
    # and http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:williams_r
    # OK
    def william_r(self, df, n_days):
        """Calculate William %R for given data.
        
        :param df: pandas.DataFrame
        :return: pandas.DataFrame
        """
        df_output = pd.DataFrame()
        
        for n in n_days:
            high = df['High'].rolling(window=n).max()
            low = df['Low'].rolling(window=n).min()
            
            william = pd.Series((high - df['Close']) / (high - low), name='William%R_' + str(n)) * -100.
            df_output = pd.concat([df_output, william], axis=1)
        
        return df_output
    
    
    # Generate difference and lag data
    # https://machinelearningmastery.com/tune-lstm-hyperparameters-keras-time-series-forecasting/    
    # OK
    def difference(self, df, lag=1):
        return df.diff(lag)
    
    def lag(self, df, l=1):    
        return df.shift(l)
    

    ############################################################################################    
    # This is the function that generates the requirement
    def generate_return(self, closing_0, closing_30):
        return (closing_30 - closing_0) / closing_0 * 100.


############################################################################################

target_variable = 'Close'
train_size = 0.6
lookahead = 30
ticker = 'BTC-USD'
epoch_count = 2048
batch_size = 2048
lookbacks = 14

stock_predictor = StockPricePredictor(ticker=ticker, seed=0)
stock_predictor.run_prediction(target_variable, train_size=train_size, lookahead=lookahead, lookbacks=lookbacks,
                               ticker=ticker, epoch_count=epoch_count, batch_size=batch_size)
