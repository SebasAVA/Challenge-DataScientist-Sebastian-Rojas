# -*- coding: utf-8 -*-
"""
Created on Sun Mar 13 21:06:42 2022

@author: sebas
"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import matplotlib
from matplotlib.backends.backend_agg import RendererAgg
import pickle
import scipy.stats as stats

reg = pickle.load(open("regression.sav", 'rb'))
clf = pickle.load(open("clf.sav", 'rb'))
scl_flt = pd.read_csv("data_streamlit.csv")
st.set_page_config(layout="wide")
#['OPERA','TIPOVUELO','temporada_alta','periodo_dia','MES']

sns.set_style('darkgrid')
matplotlib.use("agg")

_lock = RendererAgg.lock


sns.set_style('darkgrid')
row0_spacer1, row0_1, row0_spacer2, row0_2, row0_spacer3 = st.columns(
    (.1, 2, .2, 1, .1))

row0_1.title('Analyzing Delayed Flights On Santiago de Chile Airport (SCEL)')


with row0_2:
    st.write('')

row0_2.subheader(
    'A Streamlit by Sebastian Rojas Ardila')

row1_spacer1, row1_1, row1_spacer2 = st.columns((.1, 3.2, .1))

with row1_1:
    st.markdown("This is the webapp made to test the models generated on the Analysis of the Data.")
    st.markdown("You need to select the value of the variables and the web app will provide to you the expected time of delay")
    st.markdown(reg)    
row2_spacer1, row2_1, row2_spacer2, row2_2, row2_spacer3, row2_3, row2_spacer4, row2_4, row2_spacer5, row2_5 = st.columns(
    (.1, 1, .1, 1, .1, 1, .1, 1, .1, 1))
with row2_1:
    Ope = st.selectbox('Airline', scl_flt["OPERA"].value_counts().index)
with row2_2:
    TV = st.selectbox('Type of flight', scl_flt["TIPOVUELO"].value_counts().index)
with row2_3:
    Tempo = st.selectbox('Season', scl_flt["temporada_alta"].value_counts().index)
with row2_4:
    Period = st.selectbox('Period of the day', scl_flt["periodo_dia"].value_counts().index)
with row2_5:
    month = st.selectbox('Month', scl_flt["MES"].value_counts().index)
    
row3_space1, row3_1, row3_space2, row3_2, row3_space3 = st.columns(
    (.1, 3, .1, 3, .1))



with row3_1, _lock:
    
    st.subheader('Boxplot of the current Airline')
    box1, violinAtk = plt.subplots() 
    violinAtk = sns.boxplot(x="MES", y="dif_min", 
                            data=scl_flt[scl_flt["OPERA"]==Ope], showfliers = False)

    violinAtk.set_xticklabels(violinAtk.get_xticklabels(), rotation=90)

    st.pyplot(box1)

with row3_2, _lock:
    st.subheader('Boxplot of airlines based high/low season')
    box3, status = plt.subplots() 
    status = sns.boxplot(x='OPERA', y='dif_min', 
                             data=scl_flt[scl_flt["temporada_alta"]==Tempo], showfliers = False)

    status.set_xticklabels(status.get_xticklabels(), rotation=90)
    st.pyplot(box3)
    
row4_space1, row4_1, row4_space2, row4_2, row4_space3 = st.columns(
    (.1, 3, .1, 3, .1))

df2 = pd.DataFrame({'OPERA': [Ope],
                    'TIPOVUELO' : [TV],
                    'temporada_alta' : [Tempo],
                    'periodo_dia' : [Period],
                    'MES' : [month]})
ss = scl_flt[['OPERA','TIPOVUELO','temporada_alta','periodo_dia','MES']]
df = pd.concat([ss, df2], ignore_index = True, axis = 0)
df = pd.get_dummies(data=df, drop_first=True)
    
df3 = pd.DataFrame({'TIPOVUELO' : [TV],
                    'temporada_alta' : [Tempo]})
ss1 = scl_flt[['TIPOVUELO','temporada_alta']]
df1 = pd.concat([ss1, df3], ignore_index = True, axis = 0)
df1 = pd.get_dummies(data=df1, drop_first=True)    
    
with row3_1, _lock:
    st.header('Result of the regression:')
    pre = reg.predict(df.tail(1))[0]
    
    st.subheader("Time expected [Min] {:.2f}".format(pre))
    st.markdown("Probability that this flight don't get delayed more time base on the original data")
    st.subheader("Probability of {:.2f}%".format(stats.percentileofscore(scl_flt["dif_min"], pre)))
    
#with row3_2, _lock:
    #st.subheader('Result of the classification:')
    #predicted = clf.predict(df.tail(1))[0]
    #prob = clf.predict_proba(df.tail(1))
    #st.subheader(predicted)
    #st.subheader(prob)

