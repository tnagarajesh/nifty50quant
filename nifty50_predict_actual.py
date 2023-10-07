import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="NiftyQuants", layout="centered")

nifty50 = pd.read_csv('https://storage.googleapis.com/nifty50_prediction/nifty50_prediction.csv',index_col=None)
nifty50_pre = pd.read_csv('https://storage.googleapis.com/nifty50_prediction/nifty50_features_predictions.csv')

st.write("Today Nifty50 predicted close:", nifty50_pre.iat[-1,7])

st.write("Past Predicted Vs Actual Nifty50 close comparison:")

# create data
x = nifty50['Date']
z = nifty50['Actual_Nifty50_Close']
p = nifty50['Predicted_Nifty50_Close']
e = nifty50['Prediction Error (Points)']

fig, (ax1) = plt.subplots(1, 1)

ax1.plot(x, z, label="Actual Close Value", marker='o', linestyle="-")
ax1.plot(x, p, label="Predicted Close Value", marker='o', linestyle="--")
ax1.legend()
st.pyplot(fig)

st.write(nifty50)


