
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Predict Nifty50 Low", layout="wide")

nifty50 = pd.read_csv('https://storage.googleapis.com/nifty50/nifty50_prediction.csv',)
nifty50_pre = pd.read_csv('https://storage.googleapis.com/nifty50/nifty50_features_predictions_low.csv')

nifty50_pred_value = nifty50_pre.iat[-1, 4]
title = "Today Nifty50 Predicted Low Price: " + str(nifty50_pred_value)

st.header(title, divider='rainbow')

# create data
x = nifty50['Date']
z = nifty50['Actual_Nifty50_Low']
p = nifty50['Predicted_Nifty50_Low']
e = nifty50['Low Prediction Error (Points)']


fig = plt.figure(figsize=(8, 4), dpi=120)
plt.plot(x, z, label="Nifty50 Actual Low", marker="o", markersize=10, linestyle="-")
plt.plot(x, p, label="Past Nifty50 Predicted Low", marker="o", markersize=10, linestyle="--")
plt.plot("Today Nifty50 Low", nifty50_pred_value, label="Today Nifty50 Low", marker="o", markersize=10)
plt.title("Past Predicted Vs Actual Nifty50 Low")
plt.grid(visible=None, which='major', axis='y', linestyle='--')
# plt.xticks(rotation='vertical', fontsize=8)
plt.legend()
st.pyplot(fig)

st.write("")
st.dataframe(nifty50, use_container_width=True, hide_index=True)
