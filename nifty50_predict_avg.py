import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Pred Nifty50 Close", layout="wide")

nifty50_ac2_vcog = pd.read_csv('https://storage.googleapis.com/nifty50_prediction/nifty50_features_predictions_OCGV.csv')
nifty50_ac2_vcog_hog_log = pd.read_csv('https://storage.googleapis.com/nifty50_prediction/nifty50_features_predictions.csv')
nifty50_ac1_vcog = pd.read_csv('https://storage.googleapis.com/nifty50/nifty50_features_predictions_close.csv')

nifty50_ac2_pred_close_vcog = nifty50_ac2_vcog.iat[-1, 5]
nifty50_ac2_pred_close_vcog_hog_log = nifty50_ac2_vcog_hog_log.iat[-1, 7]
nifty50_ac1_pred_close_vcog = nifty50_ac1_vcog.iat[-1, 4]

nifty50_pred_value = (nifty50_ac2_pred_close_vcog + nifty50_ac2_pred_close_vcog_hog_log + nifty50_ac1_pred_close_vcog)/3

title = "Today Nifty50 Predicted Close Price: " + str(nifty50_pred_value)

st.header(title, divider='rainbow')

st.write("Model 1", nifty50_ac1_pred_close_vcog)
st.write("Model 2", nifty50_ac2_pred_close_vcog)
st.write("Model 3", nifty50_ac2_pred_close_vcog_hog_log)
