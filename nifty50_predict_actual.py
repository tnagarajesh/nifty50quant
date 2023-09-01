import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

nifty50 = pd.read_csv('https://storage.googleapis.com/nifty50/Nifty50_prediction.csv')
nifty50_pre = pd.read_csv('https://storage.googleapis.com/nifty50/nifty50_features_predictions.csv')

st.write("Today Nifty50 predicted close price:", nifty50_pre.iat[-1,7])

# create data
x = nifty50_pre['Date']
x1 = nifty50['Date']
z = nifty50['Actual_Nifty50_Close']
p = nifty50_pre['Predicted_Value']
e = nifty50['Prediction Error (Points)']

fig, (ax1, ax2) = plt.subplots(2, 1)

ax1.plot(x, z, label="Actual Close Value", marker='o', linestyle="-")
ax1.plot(x, p, label="Predicted Close Value", marker='o', linestyle="--")
ax1.legend()

ax2.plot(x1, e, label="Prediction Error", marker='o', linestyle="-")
ax2.legend()

st.pyplot(fig)
