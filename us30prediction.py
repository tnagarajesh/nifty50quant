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

US30_pre_close = US30_pre.iat[-1, 18]

x = US30['Date']
z = US30['Actual_Close']
p = US30['Predicted_Close']

st.write("US30 Prediction Close Today",US30_pre_close)

st.divider()

fig = plt.figure(figsize=(5, 2), dpi=120)
plt.plot(x, z, label="US30 Actual Close", marker="o", markersize=10, linestyle="-")
plt.plot(x, p, label="Past US30 Predicted Close", marker="o", markersize=10, linestyle="--")
plt.plot("Today US30 Predicted Close", US30_pre_close, label="Today US30 Predicted Close", marker="o", markersize=10)
plt.title("Past Predicted Vs Actual US30 Close")
plt.grid(visible=None, which='major', axis='y', linestyle='--')
plt.xticks(rotation='vertical', fontsize=8)
plt.legend()
st.pyplot(fig)


st.dataframe(US30[['Date', 'Predicted_Close', 'Actual_Close', 'Close_Prediction_Error']], hide_index=True)
st.dataframe(US30)

st.header("Model Level Metrics")

close_mae = mean_absolute_error(z, p)
 #close_mse = mean_squared_error(z, p)
 #close_r_squared = r2_score(z, p)
 #close_rmse = np.sqrt(close_mse)

st.write("**Close MAE**", close_mae)
 #st.write("**Close MSE**", close_mse)
 #st.write("**Close r2_score**", close_r_squared)
 #st.write("**Close RMSE**", close_rmse)

