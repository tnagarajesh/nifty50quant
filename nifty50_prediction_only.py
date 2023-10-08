import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Nifty50 Prediction Today", layout="wide")

nifty50_pre = pd.read_csv('https://storage.googleapis.com/nifty50_prediction/nifty50_features_predictions.csv')

nifty50_pred_value = nifty50_pre.iat[-1, 7]
title = "Today Nifty50 Predicted Close Price: " + str(nifty50_pred_value)

st.header(title, divider='rainbow')