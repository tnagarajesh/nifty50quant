import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Pred Nifty50 Close", layout="wide")

nifty50_ac2_vcog = pd.read_csv('https://storage.googleapis.com/nifty50_prediction/nifty50_features_predictions_OCGV.csv')
nifty50_ac2_vcog_hog_log = pd.read_csv('https://storage.googleapis.com/nifty50_prediction/nifty50_features_predictions.csv')
nifty50_ac1_vcog = pd.read_csv('https://storage.googleapis.com/nifty50/nifty50_features_predictions_close.csv')
nifty50_pred_avg = pd.read_csv('https://storage.googleapis.com/nifty50_prediction/nifty50_pred_avg.csv')

nifty50_ac2_pred_close_vcog = nifty50_ac2_vcog.iat[-1, 5]
nifty50_ac2_pred_close_vcog_hog_log = nifty50_ac2_vcog_hog_log.iat[-1, 7]
nifty50_ac1_pred_close_vcog = nifty50_ac1_vcog.iat[-1, 4]

nifty50_pred_value = (nifty50_ac2_pred_close_vcog + nifty50_ac2_pred_close_vcog_hog_log + nifty50_ac1_pred_close_vcog)/3

title = "Today Nifty50 Predicted Close Price (avg): " + str(nifty50_pred_value)

st.header(title, divider='rainbow')

st.write("Model 1", nifty50_ac1_pred_close_vcog)
st.write("Model 2", nifty50_ac2_pred_close_vcog)
st.write("Model 3", nifty50_ac2_pred_close_vcog_hog_log)

# create data
x = nifty50_pred_avg['Date']
z = nifty50_pred_avg['Actual_Nifty50_Close']
p = nifty50_pred_avg['Predicted_Nifty50_Close']
e = nifty50_pred_avg['Prediction Error (Points)']

#graph, table = st.columns(2)

#with graph:
fig = plt.figure(figsize=(8, 4), dpi=120)
plt.plot(x, z, label="Nifty50 Actual Close", marker="o", markersize=10, linestyle="-")
plt.plot(x, p, label="Past Nifty50 Predicted Close", marker="o", markersize=10, linestyle="--")
plt.plot("Today Nifty50 Close", nifty50_pred_value, label="Today Nifty50 Close", marker="o", markersize=10)
plt.title("Past Predicted Vs Actual Nifty50 Close")
plt.grid(visible=None, which='major', axis='y', linestyle='--')
# plt.xticks(rotation='vertical', fontsize=8)
plt.legend()
st.pyplot(fig)

st.write("")
st.dataframe(nifty50_pred_avg, use_container_width=True, hide_index=True)
