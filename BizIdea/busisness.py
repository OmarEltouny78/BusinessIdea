# -*- coding: utf-8 -*-
"""
Created on Sat Oct  1 15:00:22 2022

@author: omart
"""

import numpy as np # linear algebra
import pandas as pd

import mplsoccer

import streamlit as st

df=pd.read_csv('SalahMoshen.csv')

df1=pd.read_csv('SalahMohsenPart1.csv')

df_final=pd.concat([df,df1])

df_final.rename(columns = {'X':'x', 'Y':'y'}, inplace = True)

df_final["X"] = (100-df_final['x'])*1.05

df_final['Y']=df_final['y']*68/100

df_final['C']=abs(df_final['y']-50)*0.68

df_final["Distance"] = np.sqrt(df_final["X"]**2 +df_final["C"]**2)
df_final["Angle"] = np.where(np.arctan(7.32 * df_final["X"] / (df_final["X"]**2 + df_final["C"]**2 - (7.32/2)**2)) > 0, np.arctan(7.32 * df_final["X"] /(df_final["X"]**2 + df_final["C"]**2 - (7.32/2)**2)), np.arctan(7.32 * df_final["X"] /(df_final["X"]**2 + df_final["C"]**2 - (7.32/2)**2)) + np.pi)

#creating extra variables
df_final["X2"] = df_final['X']**2
df_final["C2"] = df_final['C']**2
df_final["AX"]  = df_final['Angle']*df_final['X']

df_final=df_final.reset_index()

df_final['Goal']=0

df_final.at[2,'Goal']=1

import pickle

loaded_model = pickle.load(open('finalized_model.sav', 'rb'))

X=df_final[['X','Y','C','Distance','Angle','AX','X2','C2']]

y=df_final[['Goal']]

df_final['xG']=loaded_model.predict_proba(X)[:,1]

#plot pitch
pitch = mplsoccer.VerticalPitch(line_color='black', half = True, pitch_type='custom', pitch_length=105, pitch_width=68, line_zorder = 2)
fig, ax = pitch.grid(grid_height=0.9, title_height=0.06, axis=False,
                     endnote_height=0.04, title_space=0, endnote_space=0)
#make heatmap
pcm = pitch.scatter(df_final['x'],df_final['y'],ax=ax['pitch'],s=(df_final['xG']*900)+100,marker='h',c='#b94b75', edgecolors='#383838')

st.dataframe(X)

st.pyplot(fig, caption='Enter any caption here')

st.video('Salah.mp4')
