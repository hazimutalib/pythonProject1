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

main_bg = "silver.png"
main_bg_ext = "png"

side_bg = "silver.png"
side_bg_ext = "png"

st.markdown(
    f"""
    <style>
    .reportview-container {{
        background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()})
    }}
   .sidebar .sidebar-content {{
        background: url(data:image/{side_bg_ext};base64,{base64.b64encode(open(side_bg, "rb").read()).decode()})
    }}
    </style>
    """,
    unsafe_allow_html=True
)

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

Branch_code = st.selectbox('Branch_code',np.sort(df['Branch_code'].unique()), key = '1')
Vehicle_Make = st.sidebar.selectbox('Vehicle_Make',np.sort(df['Vehicle_Make'].unique()), key = '2')
Year_Manufacture = st.sidebar.selectbox('Year_Manufacture', [0,1], key = '3')
Loan_Tenure = st.sidebar.selectbox('Loan_Tenure', np.sort(df['Loan_Tenure'].unique()), key = '4')
Annual_Income  = st.sidebar.selectbox('Annual_Income', np.sort(df['Annual_Income'].unique()), key = '5')
Loan_Amount = st.sidebar.number_input('Loan_Amount', key ='6')