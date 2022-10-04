# -*- coding: utf-8 -*-
"""
Created on Sat Oct  1 15:00:22 2022

@author: omart
"""

import numpy as np # linear algebra
import pandas as pd

import mplsoccer

import streamlit as st

from highlight_text import fig_text

import pickle

df=pd.read_csv('SalahMoshen.csv')

df1=pd.read_csv('SalahMohsenPart1.csv')

df_final=pd.read_csv('eventsMohsenNew - eventsMohsen.csv')

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



loaded_model = pickle.load(open('finalized_model.sav', 'rb'))

X=df_final[['X','Y','C','Distance','Angle','AX','X2','C2']]

y=df_final[['Goal']]

df_final['xG']=loaded_model.predict_proba(X)[:,1]

colors = {'Goal':'green', 'Miss':'tomato', 'Blocked':'lightblue', 'Save':'gray', 'Post':'gold'}

pitch = mplsoccer.VerticalPitch(line_color='black', half = True, pitch_type='custom', pitch_length=100, pitch_width=100, line_zorder = 3)
fig, ax = pitch.grid(grid_height=0.9, title_height=0.06, axis=False,
                     endnote_height=0.04, title_space=0, endnote_space=0)
pcm = pitch.scatter(df_final['x'],df_final['y'],ax=ax['pitch'],s=(df_final['xG']*1500)+150,marker='o',c=df_final['Result'].map(colors), edgecolors='#383838')
# Scatter plot for goals, blocked shots, missed shots

fig_text(s=f'xG: 2.145',
        x=.15, y =.15, fontsize=30,fontfamily='Andale Mono',color='black')
fig_text(s=f'Goals: 1',
        x=.45, y =.15, fontsize=30,fontfamily='Andale Mono',color='black')
fig_text(s=f'Shots: 15',
        x=.75, y =.15, fontsize=30,fontfamily='Andale Mono',color='black')
pcm = pitch.scatter(60,95,ax=ax['pitch'],marker='o',c='green', edgecolors='#383838',s=1200)
fig_text(s=f'Goal',
        x=.13, y =.275, fontsize=30,fontfamily='Andale Mono',color='black')
pcm = pitch.scatter(60,76,ax=ax['pitch'],marker='o',c='gold', edgecolors='#383838',s=1200)
fig_text(s=f'Post',
        x=.30, y =.275, fontsize=30,fontfamily='Andale Mono',color='black')
pcm = pitch.scatter(60,57,ax=ax['pitch'],marker='o',c='grey', edgecolors='#383838',s=1200)
fig_text(s=f'Save',
        x=.47, y =.275, fontsize=30,fontfamily='Andale Mono',color='black')
pcm = pitch.scatter(60,38,ax=ax['pitch'],marker='o',c='lightblue', edgecolors='#383838',s=1200)
fig_text(s=f'Blocked',
        x=.63, y =.275, fontsize=30,fontfamily='Andale Mono',color='black')
pcm = pitch.scatter(60,19,ax=ax['pitch'],marker='o',c='tomato', edgecolors='#383838',s=1200)
fig_text(s=f'Miss',
        x=.80, y =.275, fontsize=30,fontfamily='Andale Mono',color='black')

st.title('اهداف المتوقعة - صلاح محسن')

st.header('البيانات المجمعة')

st.dataframe(df_final)
st.subheader('اهداف مسجلة : 1')
st.subheader('الاهداف المتوفعة : 2.14')

st.subheader('التسديدات : 15')

st.header('خريطة التسديدات')

st.pyplot(fig)

st.subheader('هدف:اخضر')

st.subheader('فرصة ضائعة: احمر')
st.subheader('القائم: اصفر')

st.subheader('تصدي الحارس:رمادي')
st.subheader('تصدي من لاعب:لبني')


st.header('تحليل الفيديو')

st.video('Salah.mp4')
