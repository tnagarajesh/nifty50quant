import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="US30Prediction", page_icon = '✅',layout="wide")

st.title("US30 Predictions generated using machine learning algorithms")

placeholder = st.empty()

st.header("US30 Predictions Today")

placeholder = st.empty()

US30 = pd.read_csv('https://storage.googleapis.com/us30/US30_Pred_Actual.csv', index_col=None)

US30_pre = pd.read_csv('https://storage.googleapis.com/us30/US30_Pred_Features.csv')

US30_pre_CPR = pd.read_csv('https://storage.googleapis.com/us30/US30_prediction.csv')

US30_CPR = pd.read_csv('https://storage.googleapis.com/us30/us30_predict_actual.csv', index_col=None)

US30_pre_close = US30_pre.iat[-1, 18]
US30_pre_close_CPR = US30_pre_CPR.iat[-1,5]

x = US30['Date']
z = US30['Actual_Close']
p = US30['Predicted_Close']

q = US30_CPR['Predicted_Close']
r = US30_CPR['Actual_Close']

with placeholder.container():
    Close_CPR, Close_CAM = st.columns(2)
    Close_CPR.metric(label="US30-CPR Prediction Close Today", value=US30_pre_close_CPR, delta=None)
    Close_CAM.metric(label="US30-CAM Prediction Close Today", value=US30_pre_close, delta=None)

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
        st.dataframe(US30[['Date', 'Predicted_Close', 'Actual_Close', 'Close_Prediction_Error']], hide_index=True)

    Close_CPR_graph, Close_CPR_table = st.columns(2)
    with Close_CPR_graph:
        fig1 = plt.figure(figsize=(8, 4), dpi=120)
        plt.plot(x, r, label="US30 Actual Close", marker="o", markersize=10, linestyle="-")
        plt.plot(x, q, label="Past US30 Predicted Close", marker="o", markersize=10, linestyle="--")
        plt.plot("Today US30 Predicted Close CPR", US30_pre_close_CPR, label="Today US30 Predicted Close", marker="o",
                 markersize=10)
        plt.title("Past Predicted Vs Actual US30 Close")
        plt.grid(visible=None, which='major', axis='y', linestyle='--')
        plt.xticks(rotation='vertical', fontsize=8)
        plt.legend()
        st.pyplot(fig1)

    with Close_CPR_table:
        st.dataframe(US30_CPR[['Date', 'Predicted_Close', 'Actual_Close', 'Close_Prediction_Error']], hide_index=True)

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


