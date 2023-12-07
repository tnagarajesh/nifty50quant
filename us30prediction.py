import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import yfinance as yf

st.set_page_config(page_title="US30Prediction", page_icon = '✅',layout="wide")

st.title("US30 Predictions generated using machine learning algorithms")

placeholder = st.empty()

st.header("US30 Predictions Today")

placeholder = st.empty()

US30 = pd.read_csv('https://storage.googleapis.com/us30/US30_Pred_Actual.csv', index_col=None)

US30_pre = pd.read_csv('https://storage.googleapis.com/us30/US30_Pred_Features.csv')

US30_pre_CPR = pd.read_csv('https://storage.googleapis.com/us30/us30_prediction.csv')

US30_CPR = pd.read_csv('https://storage.googleapis.com/us30/us30_predict_actual.csv', index_col=None)

US30_pre_close = US30_pre.iat[-1, 18]
US30_pre_close_CPR = US30_pre_CPR.iat[-1,5]

x = US30['Date']
z = US30['Actual_Close']
p = US30['Predicted_Close']

y = US30_CPR['Date']
q = US30_CPR['Predicted_Close']
r = US30_CPR['Actual_Close']

avg_models_us30_pred = (US30_pre_close + US30_pre_close_CPR)/2

us30_history = yf.Ticker("^DJI")
us30_avg = us30_history.history(period="60d")

us30_avg['us30_avg_high_open'] = us30_avg.apply(lambda row:abs(row['High']-row['Open']),axis=1)
us30_avg['us30_avg_low_open'] = us30_avg.apply(lambda row:abs(row['Open']-row['Low']),axis=1)
us30_avg['us30_avg_close_open'] = us30_avg.apply(lambda row:abs(row['Open']-row['Close']),axis=1)
us30_avg['us30_avg_high_low_range'] = us30_avg.apply(lambda row:abs(row['High']-row['Low']),axis=1)

st.divider()

with placeholder.container():

    Close_CPR, Close_CAM, Close_Avg = st.columns(3)
    Close_CPR.metric(label="US30-CPR Prediction Close Today", value=US30_pre_close_CPR, delta=None)
    Close_CAM.metric(label="US30-CAM Prediction Close Today", value=US30_pre_close, delta=None)
    Close_Avg.metric(label="US30-Avg Prediction Close Today", value=avg_models_us30_pred, delta=None)

    st.divider()

    us30_avg_high1, us30_avg_low1, us30_avg_close1, us30_avg_range1 = st.columns(4)
    us30_avg_high1.metric(label="US30-Past 60 days Average High", value=np.mean(us30_avg['us30_avg_high_open']), delta=None)
    us30_avg_low1.metric(label="US30-Past 60 days Average Low", value=np.mean(us30_avg['us30_avg_low_open']), delta=None)
    us30_avg_close1.metric(label="US30-Past 60 days Average Close", value=np.mean(us30_avg['us30_avg_close_open']), delta=None)
    us30_avg_range1.metric(label="US30-Past 60 days Average High Low Range", value=np.mean(us30_avg['us30_avg_high_low_range']), delta=None)

    st.divider()

    us30_avg_high_graph, us30_avg_low_graph, us30_avg_close_graph, us30_avg_range_graph = st.columns(4)
    with us30_avg_high_graph:
        fig2 = plt.figure(figsize=(8, 4), dpi=120)
        plt.hist(us30_avg['us30_avg_high_open'], 5)
        st.pyplot(fig2)
    with us30_avg_low_graph:
        fig2 = plt.figure(figsize=(8, 4), dpi=120)
        plt.hist(us30_avg['us30_avg_low_open'], 5)
        st.pyplot(fig2)
    with us30_avg_close_graph:
        fig2 = plt.figure(figsize=(8, 4), dpi=120)
        plt.hist(us30_avg['us30_avg_close_open'], 5)
        st.pyplot(fig2)
    with us30_avg_range_graph:
        fig2 = plt.figure(figsize=(8, 4), dpi=120)
        plt.hist(us30_avg['us30_avg_high_low_range'], 5)
        st.pyplot(fig2)

    st.divider()

    Close_CAM_graph, Close_CAM_table = st.columns(2)
    with Close_CAM_graph:
        fig1 = plt.figure(figsize=(8, 4), dpi=120)
        plt.plot(x, z, label="US30 Actual Close", marker="o", markersize=10, linestyle="-")
        plt.plot(x, p, label="Past US30 Predicted Close", marker="o", markersize=10, linestyle="--")
        plt.plot("Today US30 Predicted Close CAM", US30_pre_close, label="Today US30 Predicted Close", marker="o",
                 markersize=10)
        plt.title("Past Predicted Vs Actual US30 Close")
        plt.grid(visible=None, which='major', axis='y', linestyle='--')
        plt.xticks(rotation='vertical', fontsize=8)
        plt.legend()
        st.pyplot(fig1)

    with Close_CAM_table:
        US301 = US30.sort_values(by='Date', ascending=False)
        st.dataframe(US301[['Date', 'Predicted_Close', 'Actual_Close', 'Close_Prediction_Error']], hide_index=True)

    Close_CPR_graph, Close_CPR_table = st.columns(2)
    with Close_CPR_graph:
        fig1 = plt.figure(figsize=(8, 4), dpi=120)
        plt.plot(y, r, label="US30 Actual Close", marker="o", markersize=10, linestyle="-")
        plt.plot(y, q, label="Past US30 Predicted Close", marker="o", markersize=10, linestyle="--")
        plt.plot("Today US30 Predicted Close CPR", US30_pre_close_CPR, label="Today US30 Predicted Close", marker="o",
                 markersize=10)
        plt.title("Past Predicted Vs Actual US30 Close")
        plt.grid(visible=None, which='major', axis='y', linestyle='--')
        plt.xticks(rotation='vertical', fontsize=8)
        plt.legend()
        st.pyplot(fig1)

    with Close_CPR_table:
        US302_CPR = US30_CPR.sort_values(by='Date', ascending=False)
        st.dataframe(US302_CPR[['Date', 'Predicted_Close', 'Actual_Close', 'Close_Prediction_Error']], hide_index=True)

    st.divider()

st.header("Model Level Metrics-CAM")

close_mae = mean_absolute_error(z, p)
 #close_mse = mean_squared_error(z, p)
 #close_r_squared = r2_score(z, p)
 #close_rmse = np.sqrt(close_mse)

st.write("**Close MAE**", close_mae)
 #st.write("**Close MSE**", close_mse)
 #st.write("**Close r2_score**", close_r_squared)
 #st.write("**Close RMSE**", close_rmse)


st.header("Model Level Metrics-CPR")

close_mae = mean_absolute_error(q, r)
 #close_mse = mean_squared_error(z, p)
 #close_r_squared = r2_score(z, p)
 #close_rmse = np.sqrt(close_mse)

st.write("**Close MAE**", close_mae)
 #st.write("**Close MSE**", close_mse)
 #st.write("**Close r2_score**", close_r_squared)
 #st.write("**Close RMSE**", close_rmse)


