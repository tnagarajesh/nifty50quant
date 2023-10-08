import pandas as pd
import streamlit as st

st.set_page_config(page_title="Nifty50 Prediction Today", layout="wide")

nifty50 = pd.read_csv('https://storage.googleapis.com/nifty50_prediction/nifty50_prediction.csv', index_col=None)

# create data
x = nifty50['Date']
z = nifty50['Actual_Nifty50_Close']
p = nifty50['Predicted_Nifty50_Close']
e = nifty50['Prediction Error (Points)']

st.write("")
st.dataframe(nifty50, use_container_width=True, hide_index=True)
