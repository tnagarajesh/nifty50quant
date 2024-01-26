import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import yfinance as yf
import datetime as dt

st.set_page_config(page_title="US30Trade", page_icon = '✅',layout="wide")

st.title("US30 Trading Dashboard")

placeholder = st.empty()

US30 = yf.Ticker("^DJI")
US30 = US30.history(period="1000d")
US30 = US30.drop(["Volume","Dividends","Stock Splits"],axis=1)

US30['High-Open'] = US30.eval('High - Open')
US30['Open-Low'] = US30.eval('Open - Low')

US30['PrevLow'] = US30['Low'].values.tolist()
US30['PrevHigh'] = US30['High'].values.tolist()
US30['PrevClose'] = US30['Close'].values.tolist()

US30["PrevLow"] = US30['PrevLow'].shift(periods=1, fill_value=0)
US30["PrevHigh"] = US30['PrevHigh'].shift(periods=1, fill_value=0)
US30["PrevClose"] = US30['PrevClose'].shift(periods=1, fill_value=0)

US30 = US30.drop([US30.index[0]])

US30['Range']=US30.eval('PrevHigh - PrevLow')
US30['H5']=US30.eval('(PrevHigh/PrevLow)*PrevClose')
US30['H4']=US30.eval('PrevClose+(Range*1.1/2)')
US30['H3']=US30.eval('PrevClose+(Range*1.1/4)')
US30['H2']=US30.eval('PrevClose+(Range*1.1/6)')
US30['H1']=US30.eval('PrevClose+(Range*1.1/12)')
US30['L1']=US30.eval('PrevClose-(Range*1.1/12)')
US30['L2']=US30.eval('PrevClose-(Range*1.1/6)')
US30['L3']=US30.eval('PrevClose-(Range*1.1/4)')
US30['L4']=US30.eval('PrevClose-(Range*1.1/2)')
US30['L5']=US30.eval('PrevClose-(H5-PrevClose)')

US30['Open_Gap']=US30.eval('Open-PrevClose')

US30['H3_L3']=US30.eval('H3-L3')

US30['PrevH3']= US30['H3'].values.tolist()
US30['PrevL3']= US30['L3'].values.tolist()

US30["PrevH3"]=US30['H3'].shift(periods=1, fill_value=0)
US30["PrevL3"]=US30['L3'].shift(periods=1, fill_value=0)

US30 = US30.drop([US30.index[0]])

def twodaypivot(H3, L3, Prev_H3,Prev_L3,H3_L3):
    if (L3 > Prev_H3):
        return 'Higher Value - Bullish'
    elif (H3 < Prev_L3):
        return "Lower Value - Bearish"
    elif ((H3 < Prev_H3) and (L3 > Prev_L3)):
        return "Inside Value - Breakout"
    elif ((H3 > Prev_H3) and (L3 < Prev_L3)):
        return "Outside Value - Sideways"
    elif (H3 > Prev_H3 and L3 < Prev_H3 and L3>Prev_L3):
        return "OH - Moderately Bullish"
    elif (H3 < Prev_H3 and L3 < Prev_L3 and H3 > Prev_L3):
        return "OL - Moderately Bearish"
    else:
        return "UV - Sideways/Breakout"

US30['twodaypivot'] = US30.apply(
    lambda row: twodaypivot(row["H3"], row["L3"], row["PrevH3"], row["PrevL3"], row["H3_L3"]), axis=1)


def market_close(open, close):
    if (close > open):
        return 'Bullish'
    else:
        return "Bearish"

US30['market_close'] = US30.apply(lambda row:market_close(row["Open"],row["Close"]),axis=1)


def Open_Level(Open, H5, H4, H3, H2, H1, L1, L2, L3, L4, L5):
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


US30['Open_Level'] = US30.apply(
    lambda row: Open_Level(row["Open"], row["H5"], row["H4"], row["H3"], row["H2"], row["H1"], row["L1"], row["L2"],
                           row["L3"], row["L4"], row["L5"]), axis=1)

US30['Return']=US30.eval('Close-Open')

US30['High_H3']=US30.eval('High-H3')
US30['High_H4']=US30.eval('High-H4')
US30['High_H5']=US30.eval('High-H5')
US30['Low_L3']=US30.eval('L3-Low')
US30['Low_L4']=US30.eval('L4-Low')
US30['Low_L5']=US30.eval('L5-Low')

US30['High_H3'] = np.maximum(US30['High_H3'], 0)
US30['High_H4'] = np.maximum(US30['High_H4'], 0)
US30['High_H5'] = np.maximum(US30['High_H5'], 0)
US30['Low_L3'] = np.maximum(US30['Low_L3'], 0)
US30['Low_L4'] = np.maximum(US30['Low_L4'], 0)
US30['Low_L5'] = np.maximum(US30['Low_L5'], 0)

US30=US30.reset_index()

def get_Day_Week(Date):
    dt = Date
    return dt.strftime('%A')

US30['Day_Week'] = US30.apply(
    lambda row: get_Day_Week(row["Date"]), axis=1)

US30['Open_H3']=US30.eval('H3-Open')
US30['Open_H4']=US30.eval('H4-Open')
US30['Open_H5']=US30.eval('H5-Open')
US30['Open_L3']=US30.eval('Open-L3')
US30['Open_L4']=US30.eval('Open-L4')
US30['Open_L5']=US30.eval('Open-L5')

US30['H5_H4']=US30.eval('H5-H4')
US30['H5_H3']=US30.eval('H5-H3')
US30['H4_H3']=US30.eval('H4-H3')

US30['L3_L4']=US30.eval('L3-L4')
US30['L3_L5']=US30.eval('L3-L5')
US30['L4_L5']=US30.eval('L4-L5')
US30['High_Close']=US30.eval('High-Close')
US30['Close_Low']=US30.eval('Close-Low')

US30.to_csv("US30_High_Analysis.csv")

st.divider()

st.header("Pivot Analysis")

Open1, H3_L3_Width, Open_Level1, Open_Gap1 = st.columns(4)
Open1.metric(label="Open", value=(US30.iloc[-1, 1]).round(2), delta=None)
H3_L3_Width.metric(label="H3 L3 Width", value=(US30.iloc[-1, 22]).round(2), delta=None)
Open_Level1.metric(label="Open Level", value=US30.iloc[-1, 27], delta=None)
Open_Gap1.metric(label="Opening Gap", value=(US30.iloc[-1, 21]).round(2), delta=None)

if (US30.iloc[-1, 1]>=US30.iloc[-2, 2]):
    open_range = "Open outside Yesterday High"
elif (US30.iloc[-1, 1]<=US30.iloc[-2, 3]):
    open_range = "Open outside Yesterday Low"
else:
 open_range = "Open within Yesterday High Low"

 twodaycam, ydayrange1, yrangeOpen = st.columns(3)
 twodaycam.metric(label="Two Day Pivot Relationship", value=US30.iloc[-1, 25], delta=None)
 ydayrange1.metric(label="Yesterday High Low Range", value=US30.iloc[-2, 10], delta=None)
 yrangeOpen.metric(label="Open vs Range", value=open_range, delta=None)

 H3L3Widthgraph, H3L3Widthrange = st.columns(2)
 with H3L3Widthgraph:
     st.write("H3 L3 Width")
     fig2 = plt.figure(figsize=(8, 4), dpi=120)
     plt.hist(US30['H3_L3'], 5)
     st.pyplot(fig2)
     plt.close()
 with H3L3Widthrange:
     st.write(US30['H3_L3'].describe())

Open_H31, Open_H41, Open_H51  = st.columns(3)
Open_H31.metric(label="Open_H3", value=(US30.iloc[-1, 36]).round(2), delta=None)
Open_H41.metric(label="Open_H4", value=(US30.iloc[-1, 37]).round(2), delta=None)
Open_H51.metric(label="Open_H5", value=(US30.iloc[-1, 38]).round(2), delta=None)

Open_L31, Open_L41, Open_L51  = st.columns(3)
Open_L31.metric(label="Open_L3", value=(US30.iloc[-1, 39]).round(2), delta=None)
Open_L41.metric(label="Open_L4", value=(US30.iloc[-1, 40]).round(2), delta=None)
Open_L51.metric(label="Open_L5", value=(US30.iloc[-1, 41]).round(2), delta=None)

H5_H41, H5_H31, H4_H31  = st.columns(3)
H5_H41.metric(label="H5_H4", value=(US30.iloc[-1, 42]).round(2), delta=None)
H5_H31.metric(label="H5_H3", value=(US30.iloc[-1, 43]).round(2), delta=None)
H4_H31.metric(label="H4_H3", value=(US30.iloc[-1, 44]).round(2), delta=None)

L3_L41, L3_L51, L4_L51  = st.columns(3)
L3_L41.metric(label="L3_L4", value=(US30.iloc[-1, 45]).round(2), delta=None)
L3_L51.metric(label="L3_L5", value=(US30.iloc[-1, 46]).round(2), delta=None)
L4_L51.metric(label="L4_L5", value=(US30.iloc[-1, 47]).round(2), delta=None)

us30_5 = pd.DataFrame()
us30_5 = US30[-10:]

xd = us30_5['Date']
oh3 = us30_5['Open_H3']
oh4 = us30_5['Open_H4']
oh5 = us30_5['Open_H5']
ol3 = us30_5['Open_L3']
ol4 = us30_5['Open_L4']
ol5 = us30_5['Open_L5']
ohigh = us30_5['High-Open']
olow = us30_5['Open-Low']
h3_l3 = us30_5['H3_L3']
H5H4 = us30_5['H5_H4']
H5H3 = us30_5['H5_H3']
H4H3 = us30_5['H4_H3']
L3L4 = us30_5['L3_L4']
L3L5 = us30_5['L3_L5']
L4L5 = us30_5['L4_L5']
highclose = us30_5['High_Close']
closelow =  us30_5['Close_Low']
rangehl = us30_5['Range']

high_open_graph, low_open_graph = st.columns(2)
with high_open_graph:
 fig1 = plt.figure(figsize=(8, 4), dpi=120)
 plt.plot(xd, ohigh, label="Open High", marker="o", markersize=10, linestyle="--")
 plt.title("Open Vs High")
 plt.grid(visible=None, which='major', axis='y', linestyle='--')
 plt.xticks(rotation='vertical', fontsize=8)
 plt.legend()
 st.pyplot(fig1)
 plt.close()
with low_open_graph:
 fig1 = plt.figure(figsize=(8, 4), dpi=120)
 plt.plot(xd, olow, label="Open Low", marker="o", markersize=10, linestyle="--")
 plt.title("Open Vs Low")
 plt.grid(visible=None, which='major', axis='y', linestyle='--')
 plt.xticks(rotation='vertical', fontsize=8)
 plt.legend()
 st.pyplot(fig1)
 plt.close()

range_graph, close_low_graph, close_high_graph = st.columns(3)
with range_graph:
  fig1 = plt.figure(figsize=(8, 4), dpi=120)
  plt.plot(xd, rangehl, label="Open Low", marker="o", markersize=10, linestyle="--")
  plt.title("High Low Range")
  plt.grid(visible=None, which='major', axis='y', linestyle='--')
  plt.xticks(rotation='vertical', fontsize=8)
  plt.legend()
  st.pyplot(fig1)
  plt.close()

with close_low_graph:
   fig1 = plt.figure(figsize=(8, 4), dpi=120)
   plt.plot(xd, closelow, label="Close Low", marker="o", markersize=10, linestyle="--")
   plt.title("Close Low Range")
   plt.grid(visible=None, which='major', axis='y', linestyle='--')
   plt.xticks(rotation='vertical', fontsize=8)
   plt.legend()
   st.pyplot(fig1)
   plt.close()

with close_high_graph:
    fig1 = plt.figure(figsize=(8, 4), dpi=120)
    plt.plot(xd, highclose, label="High Close", marker="o", markersize=10, linestyle="--")
    plt.title("High Close Range")
    plt.grid(visible=None, which='major', axis='y', linestyle='--')
    plt.xticks(rotation='vertical', fontsize=8)
    plt.legend()
    st.pyplot(fig1)
    plt.close()


st.divider()

st.header("Pivot Based Analysis")

us30_365 = pd.DataFrame()
us30_365 = US30[-365:]

us30_today_pivots_avg = us30_365[(us30_365['Open_Level'] == us30_365.iloc[-1, 27]) & (us30_365['H3_L3'] >= us30_365.iloc[-1, 22]-30) & (
                us30_365['H3_L3'] < us30_365.iloc[-1, 22] + 30) & (us30_365['twodaypivot'] == us30_365.iloc[-1, 25])]

st.write(us30_today_pivots_avg)

us30_High_H5_365, us30_High_H4_365, us30_High_H3_365, us30_Low_L3_365, us30_Low_L4_365, us30_Low_L5_365, us30_Range_365, us30_market_close, us30_week_day = st.columns(
        9)
with us30_High_H5_365:
        st.write("High_H5")
        st.write(us30_today_pivots_avg['High_H5'].describe())
        fig2 = plt.figure(figsize=(8, 4), dpi=120)
        plt.hist(us30_today_pivots_avg['High_H5'], 5)
        st.pyplot(fig2)
        plt.close()
with us30_High_H4_365:
        st.write("High_H4")
        st.write(us30_today_pivots_avg['High_H4'].describe())
        fig2 = plt.figure(figsize=(8, 4), dpi=120)
        plt.hist(us30_today_pivots_avg['High_H4'], 5)
        st.pyplot(fig2)
        plt.close()
with us30_High_H3_365:
        st.write("High_H3")
        st.write(us30_today_pivots_avg['High_H3'].describe())
        fig2 = plt.figure(figsize=(8, 4), dpi=120)
        plt.hist(us30_today_pivots_avg['High_H3'], 5)
        st.pyplot(fig2)
        plt.close()
with us30_Low_L3_365:
        st.write("Low_L3")
        st.write(us30_today_pivots_avg['Low_L3'].describe())
        fig2 = plt.figure(figsize=(8, 4), dpi=120)
        plt.hist(us30_today_pivots_avg['Low_L3'], 5)
        st.pyplot(fig2)
        plt.close()
with us30_Low_L4_365:
        st.write("Low_L4")
        st.write(us30_today_pivots_avg['Low_L4'].describe())
        fig2 = plt.figure(figsize=(8, 4), dpi=120)
        plt.hist(us30_today_pivots_avg['Low_L4'], 5)
        st.pyplot(fig2)
        plt.close()
with us30_Low_L5_365:
        st.write("Low_L5")
        st.write(us30_today_pivots_avg['Low_L5'].describe())
        fig2 = plt.figure(figsize=(8, 4), dpi=120)
        plt.hist(us30_today_pivots_avg['Low_L5'], 5)
        st.pyplot(fig2)
        plt.close()
with us30_Range_365:
        st.write("Range")
        st.write(us30_today_pivots_avg['Range'].describe())
        fig2 = plt.figure(figsize=(8, 4), dpi=120)
        plt.hist(us30_today_pivots_avg['Range'], 5)
        st.pyplot(fig2)
        plt.close()
with us30_market_close:
        st.write("Market Close")
        st.write(us30_today_pivots_avg.market_close.value_counts())
with us30_week_day:
        st.write("Week Day")
        st.write(us30_today_pivots_avg.Day_Week.value_counts())

st.divider()

st.header("Average High, Low, Close Analysis-365 days")

us30_avg_high1, us30_avg_low1, us30_avg_close1, us30_avg_range1, us30_avg_high_close, us30_avg_close_low = st.columns(6)
us30_avg_high1.metric(label="Last 365 days Average High", value=np.mean(us30_365['High-Open']).round(2), delta=None)
us30_avg_low1.metric(label="Last 365 days Average Low", value=np.mean(us30_365['Open-Low']).round(2), delta=None)
us30_avg_close1.metric(label="Last 365 days Average Close", value=np.mean(us30_365['Return']).round(2), delta=None)
us30_avg_range1.metric(label="Last 365 days Average High Low Range", value=np.mean(us30_365['Range']).round(2),delta=None)
us30_avg_high_close.metric(label="Last 365 days Average High Close Range", value=np.mean(us30_365['High_Close']).round(2),delta=None)
us30_avg_close_low.metric(label="Last 365 days Average Low Close Range",value=np.mean(us30_365['Close_Low']).round(2),delta=None)


st.divider()

us30_avg_high_graph, us30_avg_low_graph, us30_avg_close_graph, us30_avg_range_graph, us30_high_close_graph, us30_low_close_graph  = st.columns(6)
with us30_avg_high_graph:
    fig2 = plt.figure(figsize=(8, 4), dpi=120)
    plt.hist(us30_365['High-Open'], 5)
    st.pyplot(fig2)
    plt.close()
with us30_avg_low_graph:
     fig2 = plt.figure(figsize=(8, 4), dpi=120)
     plt.hist(us30_365['Open-Low'], 5)
     st.pyplot(fig2)
     plt.close()
with us30_avg_close_graph:
     fig2 = plt.figure(figsize=(8, 4), dpi=120)
     plt.hist(us30_365['Return'], 5)
     st.pyplot(fig2)
     plt.close()
with us30_avg_range_graph:
     fig2 = plt.figure(figsize=(8, 4), dpi=120)
     plt.hist(us30_365['Range'], 5)
     st.pyplot(fig2)
     plt.close()
with us30_high_close_graph:
     fig2 = plt.figure(figsize=(8, 4), dpi=120)
     plt.hist(us30_365['High_Close'], 5)
     st.pyplot(fig2)
     plt.close()

with us30_low_close_graph:
     fig2 = plt.figure(figsize=(8, 4), dpi=120)
     plt.hist(us30_365['Close_Low'], 5)
     st.pyplot(fig2)
     plt.close()

us30_High_H3, us30_High_H4, us30_High_H5, us30_Low_L3, us30_Low_L4, us30_Low_L5 = st.columns(6)
us30_High_H3.metric(label="Last 365 days Average High H3", value=np.mean(us30_365['High_H3']).round(2),delta=None)
us30_High_H4.metric(label="Last 365 days Average High H4", value=np.mean(us30_365['High_H4']).round(2),delta=None)
us30_High_H5.metric(label="Last 365 days Average High H5", value=np.mean(us30_365['High_H5']).round(2),delta=None)
us30_Low_L3.metric(label="Last 365 days Average Low L3", value=np.mean(us30_365['Low_L3']).round(2),delta=None)
us30_Low_L4.metric(label="Last 365 days Average Low L4", value=np.mean(us30_365['Low_L4']).round(2),delta=None)
us30_Low_L5.metric(label="Last 365 days Average Low L5", value=np.mean(us30_365['Low_L5']).round(2),delta=None)

st.divider()

us30_High_H3_graph, us30_High_H4_graph, us30_High_H5_graph, us30_Low_L3_graph, us30_Low_L4_graph, us30_Low_L5_graph = st.columns(6)
with us30_High_H3_graph:
    fig2 = plt.figure(figsize=(8, 4), dpi=120)
    plt.hist(us30_365['High_H3'], 5)
    st.pyplot(fig2)
    plt.close()
with us30_High_H4_graph:
    fig2 = plt.figure(figsize=(8, 4), dpi=120)
    plt.hist(us30_365['High_H4'], 5)
    st.pyplot(fig2)
    plt.close()
with us30_High_H5_graph:
    fig2 = plt.figure(figsize=(8, 4), dpi=120)
    plt.hist(us30_365['High_H5'], 5)
    st.pyplot(fig2)
    plt.close()
with us30_Low_L3_graph:
    fig2 = plt.figure(figsize=(8, 4), dpi=120)
    plt.hist(us30_365['Low_L3'], 5)
    st.pyplot(fig2)
    plt.close()
with us30_Low_L4_graph:
    fig2 = plt.figure(figsize=(8, 4), dpi=120)
    plt.hist(us30_365['Low_L4'], 5)
    st.pyplot(fig2)
    plt.close()
with us30_Low_L5_graph:
    fig2 = plt.figure(figsize=(8, 4), dpi=120)
    plt.hist(us30_365['Low_L5'], 5)
    st.pyplot(fig2)
    plt.close()

st.divider()

st.header("Average High, Low, Close Analysis-10 days")

us30_10 = pd.DataFrame()
us30_10 = US30[-10:]

us30_avg_high1_10, us30_avg_low1_10, us30_avg_close1_10, us30_avg_range1_10,  us30_avg_high_close, us30_avg_close_low  = st.columns(6)
us30_avg_high1_10.metric(label="Last 10 days Average High", value=np.mean(us30_10['High-Open']).round(2), delta=None)
us30_avg_low1_10.metric(label="Last 10 days Average Low", value=np.mean(us30_10['Open-Low']).round(2), delta=None)
us30_avg_close1_10.metric(label="Last 10 days Average Close", value=np.mean(us30_10['Return']).round(2), delta=None)
us30_avg_range1_10.metric(label="Last 10 days Average High Low Range", value=np.mean(us30_10['Range']).round(2), delta=None)
us30_avg_high_close.metric(label="Last 10 days Average High Close Range", value=np.mean(us30_10['High_Close']).round(2), delta=None)
us30_avg_close_low.metric(label="Last 10 days Average Low Close Range", value=np.mean(us30_10['Close_Low']).round(2), delta=None)

st.divider()

us30_avg_high_graph_10, us30_avg_low_graph_10, us30_avg_close_graph_10, us30_avg_range_graph_10, us30_high_close_graph, us30_low_close_graph = st.columns(6)
with us30_avg_high_graph_10:
   fig2 = plt.figure(figsize=(8, 4), dpi=120)
   plt.hist(us30_10['High-Open'], 5)
   st.pyplot(fig2)
   plt.close()
with us30_avg_low_graph_10:
   fig2 = plt.figure(figsize=(8, 4), dpi=120)
   plt.hist(us30_10['Open-Low'], 5)
   st.pyplot(fig2)
   plt.close()
with us30_avg_close_graph_10:
   fig2 = plt.figure(figsize=(8, 4), dpi=120)
   plt.hist(us30_10['Return'], 5)
   st.pyplot(fig2)
   plt.close()
with us30_avg_range_graph_10:
   fig2 = plt.figure(figsize=(8, 4), dpi=120)
   plt.hist(us30_10['Range'], 5)
   st.pyplot(fig2)
   plt.close()

with us30_high_close_graph:
   fig2 = plt.figure(figsize=(8, 4), dpi=120)
   plt.hist(us30_10['High_Close'], 5)
   st.pyplot(fig2)
   plt.close()

with us30_low_close_graph:
   fig2 = plt.figure(figsize=(8, 4), dpi=120)
   plt.hist(us30_10['Close_Low'], 5)
   st.pyplot(fig2)
   plt.close()

us30_High_H3_10, us30_High_H4_10, us30_High_H5_10, us30_Low_L3_10, us30_Low_L4_10, us30_Low_L5_10 = st.columns(6)
us30_High_H3_10.metric(label="Last 10 days Average High H3", value=np.mean(us30_10['High_H3']).round(2), delta=None)
us30_High_H4_10.metric(label="Last 10 days Average High H4", value=np.mean(us30_10['High_H4']).round(2), delta=None)
us30_High_H5_10.metric(label="Last 10 days Average High H5", value=np.mean(us30_10['High_H5']).round(2), delta=None)
us30_Low_L3_10.metric(label="Last 10 days Average Low L3", value=np.mean(us30_10['Low_L3']).round(2), delta=None)
us30_Low_L4_10.metric(label="Last 10 days Average Low L4", value=np.mean(us30_10['Low_L4']).round(2), delta=None)
us30_Low_L5_10.metric(label="Last 10 days Average Low L5", value=np.mean(us30_10['Low_L5']).round(2), delta=None)

st.divider()
us30_High_H3_graph, us30_High_H4_graph, us30_High_H5_graph, us30_Low_L3_graph, us30_Low_L4_graph, us30_Low_L5_graph = st.columns(6)
with us30_High_H3_graph:
   fig2 = plt.figure(figsize=(8, 4), dpi=120)
   plt.hist(us30_10['High_H3'], 5)
   st.pyplot(fig2)
   plt.close()
with us30_High_H4_graph:
   fig2 = plt.figure(figsize=(8, 4), dpi=120)
   plt.hist(us30_10['High_H4'], 5)
   st.pyplot(fig2)
   plt.close()
with us30_High_H5_graph:
   fig2 = plt.figure(figsize=(8, 4), dpi=120)
   plt.hist(us30_10['High_H5'], 5)
   st.pyplot(fig2)
   plt.close()
with us30_Low_L3_graph:
   fig2 = plt.figure(figsize=(8, 4), dpi=120)
   plt.hist(us30_10['Low_L3'], 5)
   st.pyplot(fig2)
   plt.close()
with us30_Low_L4_graph:
   fig2 = plt.figure(figsize=(8, 4), dpi=120)
   plt.hist(us30_10['Low_L4'], 5)
   st.pyplot(fig2)
   plt.close()
with us30_Low_L5_graph:
   fig2 = plt.figure(figsize=(8, 4), dpi=120)
   plt.hist(us30_10['Low_L5'], 5)
   st.pyplot(fig2)
   plt.close()













































































