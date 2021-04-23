import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
sns.set()
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
import shap
import base64


st.write(""" # Can You Hack It - Hong Leong Bank """)
st.write("Auto Loan Interest Rate Calculator")


data = pd.read_csv('autoloan.csv')
st.dataframe(data)

df = pd.read_csv('autoloan_super_cleaned.csv')
st.dataframe(df)


x = df.drop("Interest_Rate", axis=1)
y = df["Interest_Rate"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

cb=CatBoostRegressor(eval_metric='RMSE')
cb.fit(x_train, y_train, cat_features=[0,1,2], eval_set=(x_test, y_test),verbose = False ,plot=True)


df1 = pd.DataFrame(index=['R-Squared', 'Root Mean Squared Error'])
df1['Train Score'] = [cb.score(x_train,y_train), np.sqrt(mean_squared_error(y_train, cb.predict(x_train)))]
df1['Test Score'] = [cb.score(x_test,y_test), np.sqrt(mean_squared_error(y_test, cb.predict(x_test)))]

st.dataframe(df1)

ex = shap.TreeExplainer(cb)
shap_values = ex.shap_values(x_test)
shap.initjs()
fig, ax = plt.subplots()
shap.summary_plot(shap_values, x_test, plot_type = 'bar')
st.pyplot(fig)

fig1, ax1 = plt.subplots()
shap.summary_plot(shap_values, x_test)
st.pyplot(fig1)

def get_user_input():
    age = st.sidebar.slider('age',18,95,25)
    balance = st.sidebar.slider('balance',-10000, 90000, 0)
    day = st.sidebar.slider('day',1,31,15)
    duration = st.sidebar.slider('duration',0, 40000,100)
    campaign = st.sidebar.slider('campaign', 1,63,15)
    pdays = st.sidebar.slider('pdays',0,854,5)
    previous = st.sidebar.slider('previous',0,58,20)
    job = st.sidebar.selectbox('job', ['admin.', 'technician', 'services', 'management', 'retired',
       'blue-collar', 'unemployed', 'entrepreneur', 'housemaid',
       'unknown', 'self-employed', 'student'], key='1')
    education = st.sidebar.selectbox('education', ['secondary', 'tertiary', 'primary', 'unknown'], key='1')


    user_data = {'age':age, 'balance' : balance, 'day' :day, 'duration' : duration,
                 'campaign' : campaign, 'pdays': pdays, 'previous': previous, 'job': job,
                 'education': education }

    features = pd.DataFrame(user_data, index = [0])
    features = pd.get_dummies(features, drop_first = True)
    return features
user_input = get_user_input()

