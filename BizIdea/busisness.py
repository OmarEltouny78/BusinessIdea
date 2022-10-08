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

from scipy.spatial import distance


df=pd.read_csv('https://raw.githubusercontent.com/OmarEltouny78/BusinessIdea/main/BizIdea/eventsMohsenNew%20-%20eventsMohsen%20(1).csv')

df['action_start_x']=df.X/100
df['action_start_y']=df.Y/100
df['action1_start_x']=df.action1_start_x/100
df['action2_start_x']=df.action2_start_x/100
df['action1_start_y']=df.action1_start_y/100
df['action2_start_y']=df.action2_start_y/100

df['C']=abs(df['action_start_y']-50)*0.68

df.rename(columns={'body_part':'action_body_part_id'},inplace=True)

df["Angle"] = np.where(np.arctan(7.32 * df["action_start_x"] / (df["action_start_x"]**2 + df['C']**2 - (7.32/2)**2)) > 0, np.arctan(7.32 * df["action_start_x"] /(df["action_start_x"]**2 + df['C']**2 - (7.32/2)**2)), np.arctan(7.32 * df["action_start_x"] /(df["action_start_x"]**2 + df['C']**2 - (7.32/2)**2)) + np.pi)

goal = (1, 0.5)

# Loop over the three actions
for action in ['action', 'action1', 'action2']:
    key_start_x = '{action}_start_x'.format(action=action)
    key_start_y = '{action}_start_y'.format(action=action)
    key_start_distance = '{action}_start_distance'.format(action=action)

    df[key_start_distance] = df.apply(lambda s: distance.euclidean((s[key_start_x], s[key_start_y]), goal), axis=1)



df.rename(columns={'Goal':'action_result'},inplace=True)

# Features
columns_features = ['action_start_x', 'action_start_y', 'action_body_part_id', 'action_start_distance','action1_start_distance', 'action2_start_distance','Angle','C']

# Label: 1 if a goal, 0 otherwise
column_target = 'action_result'

X = df[columns_features]
y = df[column_target]


loaded_model = pickle.load(open('BizIdea/classifierxG.sav', 'rb'))
result = loaded_model.predict_proba(X)[:,1]
df['xG']=result

colors = {'Goal':'green', 'Miss':'tomato', 'Blocked':'lightblue', 'Save':'gray', 'Post':'gold'}

pitch = mplsoccer.VerticalPitch(line_color='black', half = True, pitch_type='custom', pitch_length=100, pitch_width=100, line_zorder = 3)
fig, ax = pitch.grid(grid_height=0.9, title_height=0.06, axis=False,
                     endnote_height=0.04, title_space=0, endnote_space=0)
pcm = pitch.scatter(df['X'],df['Y'],ax=ax['pitch'],s=(df['xG']*1500)+150,marker='o',c=df['Result'].map(colors), edgecolors='#383838')
# Scatter plot for goals, blocked shots, missed shots

fig_text(s=f'xG: ' + str(np.sum(result)),
        x=.15, y =.15, fontsize=30,fontfamily='Andale Mono',color='black')
fig_text(s=f'Goals: 1',
        x=.45, y =.15, fontsize=30,fontfamily='Andale Mono',color='black')
fig_text(s=f'Shots: '+str(len(df)),
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

df_display=df[['Mins','Secs','Result','Against','xG']]
st.title('اهداف المتوقعة - صلاح محسن')
st.header('تحليل الفيديو')

st.video('Salah.mp4')
st.header('البيانات المجمعة')

st.dataframe(df_display)
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



