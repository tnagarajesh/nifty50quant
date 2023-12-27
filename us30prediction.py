import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import yfinance as yf
from datetime import date

st.set_page_config(page_title="US30Prediction", page_icon = '✅',layout="wide")

st.title("US30 Trading Dashboard")

placeholder = st.empty()

st.header("Predictions")

placeholder = st.empty()

US30_CAM = pd.read_csv('https://storage.googleapis.com/us30/US30_Pred_Actual.csv', index_col=None)

US30_pre = pd.read_csv('https://storage.googleapis.com/us30/US30_Pred_Features.csv')

US30_pre_CPR = pd.read_csv('https://storage.googleapis.com/us30/us30_prediction.csv')

US30_CPR = pd.read_csv('https://storage.googleapis.com/us30/us30_predict_actual.csv', index_col=None)

US30_pre_close = US30_pre.iat[-1, 18]
US30_pre_close_CPR = US30_pre_CPR.iat[-1,5]

x = US30_CAM['Date']
z = US30_CAM['Actual_Close']
p = US30_CAM['Predicted_Close']

y = US30_CPR['Date']
q = US30_CPR['Predicted_Close']
r = US30_CPR['Actual_Close']

avg_models_us30_pred = (US30_pre_close + US30_pre_close_CPR)/2


US30 = yf.Ticker("^DJI")
US30 = US30.history(period="1000d")
US30 = US30.drop(["Volume","Dividends","Stock Splits"],axis=1)
US30['PrevLow']= US30['Low'].values.tolist()
US30['PrevHigh']= US30['High'].values.tolist()
US30["PrevLow"]=US30['PrevLow'].shift(periods=1, fill_value=0)
US30["PrevHigh"]=US30['PrevHigh'].shift(periods=1, fill_value=0)
US30 = US30.drop([US30.index[0]])
US30['Pivot']=US30.eval(('High+Low+Close'))/3
US30['BC']=US30.eval(('High+Low'))/2
US30['TC']=US30.eval('(Pivot-BC)+Pivot')
US30['S1']=US30.eval('(2*Pivot)-High')
US30['S2']=US30.eval('Pivot-High+Low')
US30['S3']=US30.eval('S1-(High-Low)')
US30['S4']=US30.eval('S3-(S1-S2)')
US30['R1']=US30.eval('(2*Pivot)-Low')
US30['R2']=US30.eval('Pivot+High-Low')
US30['R3']=US30.eval('R1+(High-Low)')
US30['R4']=US30.eval('R3+(R2-R1)')
US30['CPRWidth']=US30.eval('TC-BC')
US30['Prev_Pivot']= US30['Pivot'].values.tolist()
US30['Prev_BC']= US30['BC'].values.tolist()
US30['Prev_TC']= US30['TC'].values.tolist()
US30 = US30.drop([US30.index[0]])
US30['Prev_Pivot']=US30['Prev_Pivot'].shift(periods=1, fill_value=0)
US30['Prev_BC']=US30['Prev_BC'].shift(periods=1, fill_value=0)
US30['Prev_TC']=US30['Prev_TC'].shift(periods=1, fill_value=0)
US30 = US30.drop([US30.index[0]])

def twodaypivot(TC, BC, Prev_TC,Prev_BC,CPRWidth):
  if (TC > Prev_TC):
    return 'Bullish'
  elif (TC < Prev_BC):
    return "Bearish"
  elif (CPRWidth < (Prev_TC - Prev_BC)):
    return "Breakout"
  elif (CPRWidth > (Prev_TC - Prev_BC)):
    return "Sideways"
  elif (TC > Prev_TC and BC < Prev_TC):
    return "Moderately Bullish"
  elif (TC > Prev_BC and BC < Prev_BC):
    return "Moderately Bearish"
  else:
    return "Sideways"


US30['twodaypivot'] = US30.apply(
    lambda row: twodaypivot(row["TC"], row["BC"], row["Prev_TC"], row["Prev_BC"], row["CPRWidth"]), axis=1)
def market_close(Open, Close):
  if (Close > Open):
    return 'Bullish'
  else:
    return "Bearish"

US30['market_close'] = US30.apply(lambda row:market_close(row["Open"],row["Close"]),axis=1)
US30['Return']=US30.eval('Close-Open')
US30['Prev_Close']= US30['Close'].values.tolist()
US30['Prev_Close']=US30['Prev_Close'].shift(periods=1, fill_value=0)
US30 = US30.drop([US30.index[0]])
US30['Open_Gap']=US30.eval('Open-Prev_Close')
def Open_Level(Open, R4, R3, R2, R1, TC, Pivot, BC, S4, S3, S2, S1):
  if (Open > R4):
    return 'R4'
  elif (Open < R4 and Open > R3):
    return "R3"
  elif (Open < R4 and Open < R3 and Open > R2):
    return "R2"
  elif (Open < R4 and Open < R3 and Open < R2 and Open > R1):
    return "R1"
  elif (Open < R4 and Open < R3 and Open < R2 and Open < R1 and Open > TC):
    return "TC"
  elif (Open < R4 and Open < R3 and Open < R2 and Open < R1 and Open < TC and Open > Pivot):
    return "Pivot"
  elif (Open < R4 and Open < R3 and Open < R2 and Open < R1 and Open < TC and Open < Pivot and Open > BC):
    return "BC"
  elif (Open < R4 and Open < R3 and Open < R2 and Open < R1 and Open < TC and Open < Pivot and Open < BC and Open > S1):
    return "S1"
  elif (Open < R4 and Open < R3 and Open < R2 and Open < R1 and Open < TC and Open < Pivot and Open < BC and Open < S1 and Open > S2):
    return "S2"
  elif (Open < R4 and Open < R3 and Open < R2 and Open < R1 and Open < TC and Open < Pivot and Open < BC and Open < S1 and Open < S2 and Open >S3):
    return "S3"
  else:
    return "S4"


US30['Open_Level'] = US30.apply(
    lambda row: Open_Level(row["Open"], row["R4"], row["R3"], row["R2"], row["R1"], row["TC"], row["Pivot"], row["BC"],
                           row["S4"], row["S3"], row["S2"], row["S1"]), axis=1)


def Open_Level_CAM(Open, H5, H4, H3, H2, H1, L1, L2, L3, L4, L5):
    if (Open >= H5):
        return 'H5'
    elif (Open < H5 and Open >= H4):
        return "H4"
    elif (Open < H5 and Open < H4 and Open >= H3):
        return "H3"
    elif (Open < H5 and Open < H4 and Open < H3 and Open >= H2):
        return "H2"
    elif (Open < H5 and Open < H4 and Open < H3 and Open < H2 and Open >= H1):
        return "H1"
    elif (Open < H5 and Open < H4 and Open < H3 and Open < H2 and Open < H1 and Open >= L1):
        return "L1"
    elif (Open < H5 and Open < H4 and Open < H3 and Open < H2 and Open < H1 and Open < L1 and Open >= L2):
        return "L2"
    elif (Open < H5 and Open < H4 and Open < H3 and Open < H2 and Open < H1 and Open < L1 and Open < L2 and Open >= L3):
        return "L3"
    elif (
            Open < H5 and Open < H4 and Open < H3 and Open < H2 and Open < H1 and Open < L1 and Open < L2 and Open < L3 and Open >= L4):
        return "L4"
    else:
        return "L5"


US30['us30_avg_high_open'] = US30.apply(lambda row:abs(row['High']-row['Open']),axis=1)
US30['us30_avg_low_open'] = US30.apply(lambda row:abs(row['Open']-row['Low']),axis=1)
US30['us30_avg_close_open'] = US30.apply(lambda row:abs(row['Open']-row['Close']),axis=1)
US30['us30_avg_high_low_range'] = US30.apply(lambda row:abs(row['High']-row['Low']),axis=1)

US30_last_10 = US30[['us30_avg_high_open','us30_avg_low_open','us30_avg_close_open','us30_avg_high_low_range']]\
    .tail(10)

period = "1d"
interval = "5m"
us30_current_day = yf.download("^DJI", period=period, interval=interval)
us30_history = yf.Ticker("^DJI")
us30_prev_day = us30_history.history(period="2d")

pivot = (us30_prev_day.iloc[0, 1] + us30_prev_day.iloc[0, 2] + us30_prev_day.iloc[0, 3]) / 3
BCPR = (us30_prev_day.iloc[0, 1] + us30_prev_day.iloc[0, 2]) / 2
TCPR = (pivot - BCPR) + pivot

Open = us30_current_day.iloc[0, 0]

CPRWidth = TCPR - BCPR

S1 = (2 * pivot) - us30_prev_day.iloc[0, 1]
S2 = pivot - us30_prev_day.iloc[0, 1] + us30_prev_day.iloc[0, 2]
S3 = S1 - (us30_prev_day.iloc[0, 1] - us30_prev_day.iloc[0, 2])
S4 = S3 - (S1 - S2)

R1 = (2 * pivot) - us30_prev_day.iloc[0, 2]
R2 = pivot + us30_prev_day.iloc[0, 1] - us30_prev_day.iloc[0, 2]
R3 = R1 + (us30_prev_day.iloc[0, 1] - us30_prev_day.iloc[0, 2])
R4 = R3 + (R2 - R1)

Range = us30_prev_day.iloc[0, 1] - us30_prev_day.iloc[0, 2]
H5 = (us30_prev_day.iloc[0, 1]/us30_prev_day.iloc[0, 2])*us30_prev_day.iloc[0, 3]
H4 = us30_prev_day.iloc[0, 3]+(Range*1.1/2)
H3 = us30_prev_day.iloc[0, 3]+(Range*1.1/4)
H2 = us30_prev_day.iloc[0, 3]+(Range*1.1/6)
H1 = us30_prev_day.iloc[0, 3]+(Range*1.1/12)
L1 = us30_prev_day.iloc[0, 3]-(Range*1.1/12)
L2 = us30_prev_day.iloc[0, 3]-(Range*1.1/6)
L3 = us30_prev_day.iloc[0, 3]-(Range*1.1/4)
L4 = us30_prev_day.iloc[0, 3]-(Range*1.1/2)
L5 = us30_prev_day.iloc[0, 3]-(H5-us30_prev_day.iloc[0, 3])

H3_L3 = H3-L3

Open_Level = Open_Level(Open, R4, R3, R2, R1, TCPR, pivot, BCPR, S4, S3, S2, S1)

Open_Gap = Open - us30_prev_day.iloc[0, 3]

Open_Level_CAM1 = Open_Level_CAM(Open, H5, H4, H3, H2, H1, L1, L2, L3, L4, L5)

# Create a dictionary with keys 'Timestamp', 'String', and 'Number' and the corresponding lists as its values
#features = [{'Date': date.today(), 'Open': Open, 'CPRWidth': CPRWidth, 'Open_Level': Open_Level, 'Open_Gap': Open_Gap}]

# Create a pandas DataFrame from the dictionary
#us30_features = pd.DataFrame(features)

st.divider()

with placeholder.container():

    Close_CPR, Close_CAM, Close_Avg = st.columns(3)
    Close_CPR.metric(label="US30-CPR Prediction Close Today", value=US30_pre_close_CPR.round(2), delta=None)
    Close_CAM.metric(label="US30-CAM Prediction Close Today", value=US30_pre_close.round(2), delta=None)
    Close_Avg.metric(label="US30-Avg Prediction Close Today", value=avg_models_us30_pred.round(2), delta=None)

    st.divider()

    st.header("Pivot Analysis")

    Open1, CPRWidth1, H3_L3_Width, Open_Level1, Open_Gap1, CAM_Open_Level = st.columns(6)
    Open1.metric(label="Open", value=Open.round(2), delta=None)
    CPRWidth1.metric(label="CPR Width", value=CPRWidth.round(2), delta=None)
    H3_L3_Width.metric(label="H3 L3 Width", value=H3_L3.round(2), delta=None)
    Open_Level1.metric(label="Open Level", value=Open_Level, delta=None)
    Open_Gap1.metric(label="Opening Gap", value=Open_Gap.round(2), delta=None)
    CAM_Open_Level.metric(label="CAM Open Level", value=Open_Level_CAM1, delta=None)

    st.divider()

    st.header("Average High, Low, Close Analysis")

    us30_avg_high1, us30_avg_low1, us30_avg_close1, us30_avg_range1 = st.columns(4)
    us30_avg_high1.metric(label="Last 1000 days Average High", value=np.mean(US30['us30_avg_high_open']).round(2), delta=None)
    us30_avg_low1.metric(label="Last 1000 days Average Low", value=np.mean(US30['us30_avg_low_open']).round(2), delta=None)
    us30_avg_close1.metric(label="Last 1000 days Average Close", value=np.mean(US30['us30_avg_close_open']).round(2), delta=None)
    us30_avg_range1.metric(label="Last 1000 days Average High Low Range", value=np.mean(US30['us30_avg_high_low_range']).round(2), delta=None)

    st.divider()

    us30_avg_high_graph, us30_avg_low_graph, us30_avg_close_graph, us30_avg_range_graph = st.columns(4)
    with us30_avg_high_graph:
        fig2 = plt.figure(figsize=(8, 4), dpi=120)
        plt.hist(US30['us30_avg_high_open'], 5)
        st.pyplot(fig2)
    with us30_avg_low_graph:
        fig2 = plt.figure(figsize=(8, 4), dpi=120)
        plt.hist(US30['us30_avg_low_open'], 5)
        st.pyplot(fig2)
    with us30_avg_close_graph:
        fig2 = plt.figure(figsize=(8, 4), dpi=120)
        plt.hist(US30['us30_avg_close_open'], 5)
        st.pyplot(fig2)
    with us30_avg_range_graph:
        fig2 = plt.figure(figsize=(8, 4), dpi=120)
        plt.hist(US30['us30_avg_high_low_range'], 5)
        st.pyplot(fig2)

    st.divider()

    pivot_10_high, pivot_10_low, pivot_10_close, pivot_10_range = st.columns(4)
    pivot_10_high.metric(label="Average 10 Days High", value=np.mean(US30_last_10['us30_avg_high_open']).round(2), delta=None)
    pivot_10_low.metric(label="Average 10 Days Low", value=np.mean(US30_last_10['us30_avg_low_open']).round(2), delta=None)
    pivot_10_close.metric(label="Average 10 Days Close", value=np.mean(US30_last_10['us30_avg_close_open']).round(2), delta=None)
    pivot_10_range.metric(label="Average 10 Days High Low Range", value=np.mean(US30_last_10['us30_avg_high_low_range']).round(2), delta=None)

    pivot_10_high_graph, pivot_10_low_graph, pivot_10_close_graph, pivot_10_range_graph = st.columns(4)
    with pivot_10_high_graph:
        fig2 = plt.figure(figsize=(8, 4), dpi=120)
        plt.hist(US30_last_10['us30_avg_high_open'], 5)
        st.pyplot(fig2)
    with pivot_10_low_graph:
        fig2 = plt.figure(figsize=(8, 4), dpi=120)
        plt.hist(US30_last_10['us30_avg_low_open'], 5)
        st.pyplot(fig2)
    with pivot_10_close_graph:
        fig2 = plt.figure(figsize=(8, 4), dpi=120)
        plt.hist(US30_last_10['us30_avg_close_open'], 5)
        st.pyplot(fig2)
    with pivot_10_range_graph:
        fig2 = plt.figure(figsize=(8, 4), dpi=120)
        plt.hist(US30_last_10['us30_avg_high_low_range'], 5)
        st.pyplot(fig2)

    st.divider()

    st.header("Trade - Entry")
    Entry_High, Entry_Low, H3_Open, H4_Open, L3_Open, L4_Open = st.columns(6)
    Entry_High.metric(label="Short Entry Price", value=Open+np.mean(US30_last_10['us30_avg_high_open']).round(2),delta=None)
    Entry_Low.metric(label="Long Entry Price", value=Open-np.mean(US30_last_10['us30_avg_low_open']).round(2),delta=None)
    H3_Open.metric(label="H3 Open Gap",value=(H3-Open).round(2),delta=None)
    H4_Open.metric(label="H4 Open Gap", value=(H4-Open).round(2),delta=None)
    L3_Open.metric(label="L3 Open Gap",value=(Open-L3).round(2),delta=None)
    L4_Open.metric(label="L4 Open Gap", value=(Open-L4).round(2),delta=None)


    R11,R21,R31,R41 = st.columns(4)
    R11.metric(label="R1",value=R1.round(2),delta=None)
    R21.metric(label="R2",value=R2.round(2),delta=None)
    R31.metric(label="R2",value=R3.round(2),delta=None)
    R41.metric(label="R2",value=R4.round(2),delta=None)

    H61, H31, H41, H51 = st.columns(4)
    H61.metric(label="H2", value=H5.round(2), delta=None)
    H31.metric(label="H3", value=H3.round(2), delta=None)
    H41.metric(label="H4", value=H4.round(2), delta=None)
    H51.metric(label="H5", value=H5.round(2), delta=None)


    S41, S11,S21,S31 = st.columns(4)
    S41.metric(label="S4", value=S4.round(2), delta=None)
    S11.metric(label="S1",value=S1.round(2),delta=None)
    S21.metric(label="S2",value=S2.round(2),delta=None)
    S31.metric(label="S3",value=S3.round(2),delta=None)

    L31, L41, L51, L61 = st.columns(4)
    L61.metric(label="L2", value=L2.round(2), delta=None)
    L31.metric(label="L3", value=L3.round(2), delta=None)
    L41.metric(label="L4", value=L4.round(2), delta=None)
    L51.metric(label="L5", value=L5.round(2), delta=None)



    sx = np.array(["Short Entry", "R1", "H3", "H4", "R2", "R3", "R4", "H5"])
    sy = np.array([Open + np.mean(US30_last_10['us30_avg_high_open']), R1, H3, H4, R2, R3, R4, H5])

    lx = np.array(["Long Entry", "S1", "L3", "S2", "L4", "S3", "S4", "L5"])
    ly = np.array([Open-np.mean(US30_last_10['us30_avg_low_open']), S1, L3, S2, L4, S3, S4, L5])

    short_entry_graph, long_entry_graph = st.columns(2)
    with short_entry_graph:
        fig1 = plt.figure(figsize=(8, 4), dpi=120)
        plt.bar(sx, sy, )
        plt.xticks(rotation='vertical', fontsize=8)
        plt.ylim(35000, 39000)
        st.pyplot(fig1)

    with long_entry_graph:
        fig1 = plt.figure(figsize=(8, 4), dpi=120)
        plt.bar(lx, ly, )
        plt.xticks(rotation='vertical', fontsize=8)
        plt.ylim(35000, 39000)
        st.pyplot(fig1)



    st.divider()

    st.header("Model Performance Metrics")

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
        US301 = US30_CAM.sort_values(by='Date', ascending=False)
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

st.header("Model Level Metrics-CAM")
close_mae = mean_absolute_error(z, p)
close_r_squared = r2_score(z, p)

st.write("**Close MAE**", close_mae)
st.write("**Close r2_score**", close_r_squared)


st.header("Model Level Metrics-CPR")
close_mae = mean_absolute_error(q, r)
close_r_squared = r2_score(z, p)

st.write("**Close MAE**", close_mae)
st.write("**Close r2_score**", close_r_squared)



