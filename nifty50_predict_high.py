import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Pred Nifty50 High", layout="wide")

nifty50 = pd.read_csv('https://storage.googleapis.com/nifty50/nifty50_prediction.csv',)
nifty50_pre = pd.read_csv('https://storage.googleapis.com/nifty50/nifty50_features_predictions_high.csv')

nifty50_pred_value = nifty50_pre.iat[-1, 4]
title = "Today Nifty50 Predicted High Price: " + str(nifty50_pred_value)

st.header(title, divider='rainbow')

# create data
x = nifty50['Date']
z = nifty50['Actual_Nifty50_High']
p = nifty50['Predicted_Nifty50_High']
e = nifty50['High Prediction Error (Points)']


fig = plt.figure(figsize=(8, 4), dpi=120)
plt.plot(x, z, label="Nifty50 Actual High", marker="o", markersize=10, linestyle="-")
plt.plot(x, p, label="Past Nifty50 Predicted High", marker="o", markersize=10, linestyle="--")
plt.plot("Today Nifty50 High", nifty50_pred_value, label="Today Nifty50 High", marker="o", markersize=10)
plt.title("Past Predicted Vs Actual Nifty50 High")
plt.grid(visible=None, which='major', axis='y', linestyle='--')
# plt.xticks(rotation='vertical', fontsize=8)
plt.legend()
st.pyplot(fig)

# Create a pandas DataFrame from the dictionary
nifty50_high = pd.DataFrame()

nifty50_high['Date'] = x
nifty50_high['Actual_Nifty50_High'] = z
nifty50_high['Predicted_Nifty50_High'] = p
nifty50_high['High Prediction Error (Points)'] = e

st.write("")

st.dataframe(nifty50_high, use_container_width=True, hide_index=True)

