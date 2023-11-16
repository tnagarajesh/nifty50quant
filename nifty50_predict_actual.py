import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Nifty50 Prediction Today", layout="wide")

nifty50 = pd.read_csv('https://storage.googleapis.com/nifty50/nifty50_predict_actual.csv', index_col=None)

nifty50_pre = pd.read_csv('https://storage.googleapis.com/nifty50/nifty50_prediction.csv')
nifty50_pred_high = pd.read_csv('https://storage.googleapis.com/nifty50/nifty50_prediction_high.csv')
nifty50_pred_low= pd.read_csv('https://storage.googleapis.com/nifty50/nifty50_prediction_low.csv')

nifty50_pre_close = nifty50_pre.iat[-1, 5]
nifty50_pre_high = nifty50_pred_high.iat[-1, 5]
nifty50_pre_low = nifty50_pred_low.iat[-1, 5]

title = "Today Nifty50 Predictions: Close-" + str(nifty50_pre_close) + " High-" + str(nifty50_pre_high) + " Low-" + str(nifty50_pre_low)

st.header(title, divider='rainbow')

# create data
x = nifty50['Date']
z = nifty50['Actual_Close']
p = nifty50['Predicted_Close']

q = nifty50['Predicted_High']
r = nifty50['Actual_High']

s = nifty50['Predicted_Low']
t = nifty50['Actual_Low']

close_graph, high_graph, low_graph = st.columns(3)

with close_graph:
    fig = plt.figure(figsize=(8, 4), dpi=120)
    plt.plot(x, z, label="Nifty50 Actual Close", marker="o", markersize=10, linestyle="-")
    plt.plot(x, p, label="Past Nifty50 Predicted Close", marker="o", markersize=10, linestyle="--")
    plt.plot("Today Nifty50 Predicted Close", nifty50_pre_close, label="Today Nifty50 Predicted Close", marker="o", markersize=10)
    plt.title("Past Predicted Vs Actual Nifty50 Close")
    plt.grid(visible=None, which='major', axis='y', linestyle='--')
    plt.xticks(rotation='vertical', fontsize=8)
    plt.legend()
    st.pyplot(fig)

with high_graph:
    fig1 = plt.figure(figsize=(8, 4), dpi=120)
    plt.plot(x, r, label="Nifty50 Actual High", marker="o", markersize=10, linestyle="-")
    plt.plot(x, q, label="Past Nifty50 Predicted High", marker="o", markersize=10, linestyle="--")
    plt.plot("Today Nifty50 Predicted High", nifty50_pre_high, label="Today Nifty50 High", marker="o", markersize=10)
    plt.title("Past Predicted Vs Actual Nifty50 High")
    plt.grid(visible=None, which='major', axis='y', linestyle='--')
    plt.xticks(rotation='vertical', fontsize=8)
    plt.legend()
    st.pyplot(fig1)

with low_graph:
    fig2 = plt.figure(figsize=(8, 4), dpi=120)
    plt.plot(x, t, label="Nifty50 Actual Low", marker="o", markersize=10, linestyle="-")
    plt.plot(x, s, label="Past Nifty50 Predicted Low", marker="o", markersize=10, linestyle="--")
    plt.plot("Today Nifty50 Predicted Low", nifty50_pre_low, label="Today Nifty50 Low", marker="o", markersize=10)
    plt.title("Past Predicted Vs Actual Nifty50 Low")
    plt.grid(visible=None, which='major', axis='y', linestyle='--')
    plt.xticks(rotation='vertical', fontsize=8)
    plt.legend()
    st.pyplot(fig2)

st.write("")
st.dataframe(nifty50, use_container_width=True, hide_index=True)
