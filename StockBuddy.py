from json import load
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import date
from dateutil.relativedelta import relativedelta
import streamlit as st
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import LSTM
from keras.layers import Dense, Dropout
from sklearn.preprocessing import MinMaxScaler

START = date.today() - relativedelta(years=10)
END = date.today().strftime("%Y-%m-%d")

st.title("Stock Trend Prediction")

stocks = ("GOOG","MSFT","TSLA","AMZN","IBM","INTC","005930.KS")
user_input = st.selectbox("Select the stock for prediction", stocks)

df = yf.download(user_input, START, END)

#Future Days to predict
future_days = st.slider("Number of future days for prediction", 1, 365)

#Describing Data
st.subheader('Data of last 10 years')
st.write(df.describe())

#Visualizations
st.subheader('Closing Price vs Time Graph')
fig = plt.figure(figsize=(12,6))
plt.ylabel('Closing price in USD')
plt.xlabel('Time')
plt.plot(df.Close)
st.pyplot(fig)


st.subheader('Closing Price vs Time Graph with 100MA')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize=(12,6))
plt.ylabel('Closing price in USD')
plt.xlabel('Time')
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)


st.subheader('Closing Price vs Time Graph with 100MA & 200MA')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize=(12,6))
plt.ylabel('Closing price in USD')
plt.xlabel('Time')
plt.plot(ma100)
plt.plot(ma200)
plt.plot(df.Close)
st.pyplot(fig)

df1=df.reset_index()['Close']

#Scaling
scaler=MinMaxScaler(feature_range=(0,1))
df1=scaler.fit_transform(np.array(df1).reshape(-1,1))

##splitting dataset into train and test split
training_size=int(len(df1)*0.65)
test_size=len(df1)-training_size
train_data,test_data=df1[0:training_size,:],df1[training_size:len(df1),:1]

# convert an array of values into a dataset matrix
def create_dataset(dataset, time_step=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-time_step-1):
		a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100 
		dataX.append(a)
		dataY.append(dataset[i + time_step, 0])
	return np.array(dataX), np.array(dataY)

# reshape into X=t,t+1,t+2,t+3 and Y=t+4
time_step = 100
X_train, y_train = create_dataset(train_data, time_step)
X_test, ytest = create_dataset(test_data, time_step)

# reshape input to be [samples, time steps, features] which is required for LSTM
X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)

#Load my Model
if user_input == "AMZN":
    model = load_model('Stock_Models/LSTM_AMZN.h5')
elif user_input == "MSFT":
    model = load_model('Stock_Models/LSTM_MSFT.h5')
elif user_input == "TSLA":
    model = load_model('Stock_Models/LSTM_TSLA.h5')
elif user_input == "GOOG":
    model = load_model('Stock_Models/LSTM_GOOG.h5')
elif user_input == "IBM":
    model = load_model('Stock_Models/LSTM_IBM.h5')
elif user_input == "INTC":
    model = load_model('Stock_Models/LSTM_INTC.h5')
elif user_input == "005930.KS":
    model = load_model('Stock_Models/LSTM_SAMSUNG.h5')


### Lets Do the prediction and check performance metrics
train_predict=model.predict(X_train)
test_predict=model.predict(X_test)

##Transformback to original form
train_predict=scaler.inverse_transform(train_predict)
test_predict=scaler.inverse_transform(test_predict)

x_input=test_data[(len(test_data)-100):].reshape(1,-1)

temp_input=list(x_input)
temp_input=temp_input[0].tolist()

# demonstrate prediction for next 10 days
from numpy import array

lst_output=[]
n_steps=100
i=0
while(i<future_days):
    
    if(len(temp_input)>100):
        #print(temp_input)
        x_input=np.array(temp_input[1:])
        print("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
        #print(x_input)
        yhat = model.predict(x_input, verbose=0)
        print("{} day output {}".format(i,yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
        #print(temp_input)
        lst_output.extend(yhat.tolist())
        i=i+1
    else:
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        print(yhat[0])
        temp_input.extend(yhat[0].tolist())
        print(len(temp_input))
        lst_output.extend(yhat.tolist())
        i=i+1

#day_new=np.arange(1,101)
day_pred=np.arange(0,future_days)

st.subheader('Predicted data of next '+str(future_days)+' days')
fig2 = plt.figure(figsize=(12,6))
plt.ylabel('Predicted price in USD')
plt.xlabel('Number of days of prediction')
plt.plot(day_pred,scaler.inverse_transform(lst_output),'y')
st.pyplot(fig2)


df3=df1.tolist()
df3.extend(lst_output)
df3=scaler.inverse_transform(df3).tolist()

st.subheader('Combined data of original and predicted price')
fig3 = plt.figure(figsize=(12,6))
plt.ylabel('Predicted price in USD')
plt.xlabel('Number of days')
plt.plot(df3)
st.pyplot(fig3)

