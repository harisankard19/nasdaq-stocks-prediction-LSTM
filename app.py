import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow
from tensorflow.keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt


model = load_model(models\stock_pred.keras)


st.header("Nasdaq Stock Price Predictor 2025")

st.markdown(
    "This app uses historical stock data and an LSTM model to predict future stock prices. "
    "It pulls data from Yahoo Finance and visualizes results interactively."
)

stock = st.text_input("Enter the Stock Symbol Below(Eg. NVDA for Nvidia): ", "NVDA")
start = '2014-01-01'
end = '2025-05-01'

data = yf.download(stock, start, end)

st.subheader('Stock Data')
st.write(data)






data_train = data.Close[0 : int(len(data)*0.80)]           # First 80%
data_test  = data.Close[int(len(data)*0.80) : ]            # Remaining 20%

from  sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

past_100_days = data_train.tail(100)
data.test = pd.concat([past_100_days, data_test], ignore_index=True)
data_test_scale = scaler.fit_transform(data_test)

#50 days ma:
st.subheader('Price vs MA50')
ma_50_days = data.Close.rolling(50).mean()
fig1 = plt.figure(figsize=(10,8))
plt.plot(ma_50_days, 'r')
plt.plot(data.Close, 'g')
plt.legend()
plt.show()
st.pyplot(fig1)

#100 days ma:
st.subheader('Price vs MA50 vs MA100')
ma_100_days = data.Close.rolling(100).mean()
fig2 = plt.figure(figsize=(10,8))
plt.plot(ma_50_days, 'b')
plt.plot(ma_100_days, 'r')
plt.plot(data.Close, 'g')

plt.legend()
plt.show()
st.pyplot(fig2)

#200 days ma:
st.subheader('Price vs MA100 vs MA200')
ma_200_days = data.Close.rolling(200).mean()
fig3 = plt.figure(figsize=(10,8))
plt.plot(ma_100_days, 'b')
plt.plot(ma_200_days, 'r')
plt.plot(data.Close, 'g')
plt.legend()
plt.show()
st.pyplot(fig3)





X = []
y = []

for i in range(100, data_test_scale.shape[0]):
    X.append(data_test_scale[i-100:i])
    y.append(data_test_scale[i,0])

X,y = np.array(X), np.array(y)



#predict

predict = model.predict(X)


scale = 1/scaler.scale_
predict = predict * scale #y_hat
y = y * scale #y


st.subheader('Original Price vs Predicted Price')
fig4 = plt.figure(figsize=(10,8))
plt.plot(predict, 'r', label = "PRED Price")
plt.plot(y, 'g', label = "OG Price")

plt.legend()
plt.show()
st.pyplot(fig4)
