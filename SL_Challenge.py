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
import matplotlib.image as mping

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
        
row2_spacer1, row2_1, row2_spacer2, row2_2, row2_spacer3, row2_3, row2_spacer4, row2_4, row2_spacer5, row2_5 = st.columns(
    (.1, 1, .1, 1, .1, 1, .1, 1, .1, 1))
with row2_1:
    skill_sel = st.selectbox('Airline', scl_flt["OPERA"].value_counts().index)
with row2_2:
    skill_sel1 = st.selectbox('Type of flight', scl_flt["TIPOVUELO"].value_counts().index)
with row2_3:
    skill_sel2 = st.selectbox('Season', scl_flt["temporada_alta"].value_counts().index)
with row2_4:
    skill_sel3 = st.selectbox('Period of the day', scl_flt["periodo_dia"].value_counts().index)
with row2_5:
    skill_sel4 = st.selectbox('Month', scl_flt["MES"].value_counts().index)