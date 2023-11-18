import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="NiftyPrediction", page_icon = '✅',layout="wide")

st.title("Nifty50 Predictions generated using machine learning algorithms")

placeholder = st.empty()

st.header("Nifty50 Predictions Today",divider='rainbow')

placeholder = st.empty()

nifty50 = pd.read_csv('https://storage.googleapis.com/nifty50/nifty50_predict_actual.csv', index_col=None)

nifty50.to_csv("nifty50_predict_actual.csv")

nifty50_pre = pd.read_csv('https://storage.googleapis.com/nifty50/nifty50_prediction.csv')
nifty50_pred_high = pd.read_csv('https://storage.googleapis.com/nifty50/nifty50_prediction_high.csv')
nifty50_pred_low= pd.read_csv('https://storage.googleapis.com/nifty50/nifty50_prediction_low.csv')

nifty50_pre_close = nifty50_pre.iat[-1, 5]
nifty50_pre_high = nifty50_pred_high.iat[-1, 5]
nifty50_pre_low = nifty50_pred_low.iat[-1, 5]

x = nifty50['Date']
z = nifty50['Actual_Close']
p = nifty50['Predicted_Close']

q = nifty50['Predicted_High']
r = nifty50['Actual_High']

s = nifty50['Predicted_Low']
t = nifty50['Actual_Low']

with placeholder.container():
    High, Low, Close = st.columns(3)
    High.metric(label="Nifty50 Predicted High", value=nifty50_pre_high, delta=None)
    Low.metric(label="Nifty50 Predicted Low", value=nifty50_pre_low, delta=None)
    Close.metric(label="Nifty50 Predicted Close", value=nifty50_pre_close, delta=None)

    st.divider()

    High_graph, High_table = st.columns(2)
    with High_graph:
        fig1 = plt.figure(figsize=(8, 4), dpi=120)
        plt.plot(x, r, label="Nifty50 Actual High", marker="o", markersize=10, linestyle="-")
        plt.plot(x, q, label="Past Nifty50 Predicted High", marker="o", markersize=10, linestyle="--")
        plt.plot("Today Nifty50 Predicted High", nifty50_pre_high, label="Today Nifty50 High", marker="o", markersize=10)
        plt.title("Past Predicted Vs Actual Nifty50 High")
        plt.grid(visible=None, which='major', axis='y', linestyle='--')
        plt.xticks(rotation='vertical', fontsize=8)
        plt.legend()
        st.pyplot(fig1)

    with High_table:
        st.dataframe(nifty50[['Date', 'Predicted_High', 'Actual_High', 'High_ Prediction_Error']], hide_index=True)

    st.divider()

    Low_graph, Low_table = st.columns(2)
    with Low_graph:
        fig2 = plt.figure(figsize=(8, 4), dpi=120)
        plt.plot(x, t, label="Nifty50 Actual Low", marker="o", markersize=10, linestyle="-")
        plt.plot(x, s, label="Past Nifty50 Predicted Low", marker="o", markersize=10, linestyle="--")
        plt.plot("Today Nifty50 Predicted Low", nifty50_pre_low, label="Today Nifty50 Low", marker="o", markersize=10)
        plt.title("Past Predicted Vs Actual Nifty50 Low")
        plt.grid(visible=None, which='major', axis='y', linestyle='--')
        plt.xticks(rotation='vertical', fontsize=8)
        plt.legend()
        st.pyplot(fig2)
    with Low_table:
        st.dataframe(nifty50[['Date', 'Predicted_Low', 'Actual_Low', 'Low_Prediction_Error']], hide_index=True)

    st.divider()

    Close_graph, Close_table = st.columns(2)
    with Close_graph:
        fig = plt.figure(figsize=(8, 4), dpi=120)
        plt.plot(x, z, label="Nifty50 Actual Close", marker="o", markersize=10, linestyle="-")
        plt.plot(x, p, label="Past Nifty50 Predicted Close", marker="o", markersize=10, linestyle="--")
        plt.plot("Today Nifty50 Predicted Close", nifty50_pre_close, label="Today Nifty50 Predicted Close", marker="o", markersize=10)
        plt.title("Past Predicted Vs Actual Nifty50 Close")
        plt.grid(visible=None, which='major', axis='y', linestyle='--')
        plt.xticks(rotation='vertical', fontsize=8)
        plt.legend()
        st.pyplot(fig)
    with Close_table:
        st.dataframe(nifty50[['Date', 'Predicted_Close', 'Actual_Close', 'Close_Prediction_Error']], hide_index=True)

    st.header("Trade Level Analytics", divider='rainbow')

    st.write("**Till date cumulative returns(1 Lot):** ", (np.sum([nifty50['Net_ Trade_Profit_Loss']]))*50)

    st.dataframe(nifty50[['Date', 'Short_Entry_Price', 'Short_Exit_Price', 'Short_Trade_Profit_Loss', 'Long_Entry_Price', 'Long_Exit_Price','Long_Trade_Profit_Loss', 'Net_ Trade_Profit_Loss']], hide_index=True)

    st.header("Model Level Metrics", divider='rainbow')

    high_metrics, low_metrics, close_metrics = st.columns(3)
    with high_metrics:
        high_mae = mean_absolute_error(r, q)
        #high_mse = mean_squared_error(r, q)
        #high_r_squared = r2_score(r, q)
        #high_rmse = np.sqrt(high_mse)

        st.write("**High MAE**",high_mae)
        #st.write("**High MSE**", high_mse)
        #st.write("**High r2_score**",high_r_squared)
        #st.write("**High RMSE**", high_rmse)

    with low_metrics:
        low_mae = mean_absolute_error(t, s)
        #low_mse = mean_squared_error(t, s)
        # low_r_squared = r2_score(t, s)
        #low_rmse = np.sqrt(low_mse)

        st.write("**Low MAE**", low_mae)
        #st.write("**Low MSE**", low_mse)
        #st.write("**Low r2_score**", low_r_squared)
        #st.write("**Low RMSE**", low_rmse)

    with close_metrics:
        close_mae = mean_absolute_error(z, p)
        #close_mse = mean_squared_error(z, p)
        # close_r_squared = r2_score(z, p)
        #close_rmse = np.sqrt(close_mse)

        st.write("**Close MAE**", close_mae)
        #st.write("**Close MSE**", close_mse)
        # st.write("**Close r2_score**", close_r_squared)
        #st.write("**Close RMSE**", close_rmse)

