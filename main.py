import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
sns.set()
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
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

df = pd.get_dummies(df, drop_first = True)

x = df.drop("Interest_Rate", axis=1)
y = df["Interest_Rate"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

xgb=XGBRegressor(max_depth = 4)
xgb.fit(x_train, y_train)

df1 = pd.DataFrame(index=['R-Squared', 'Root Mean Squared Error'])
df1['Train Score'] = [xgb.score(x_train,y_train), np.sqrt(mean_squared_error(y_train, xgb.predict(x_train)))]
df1['Test Score'] = [xgb.score(x_test,y_test), np.sqrt(mean_squared_error(y_test, xgb.predict(x_test)))]

st.dataframe(df1)


ex = shap.TreeExplainer(xgb)
shap_values = ex.shap_values(x_test)
shap.initjs()
fig, ax = plt.subplots()
shap.summary_plot(shap_values, x_test, plot_type = 'bar')
st.pyplot(fig)

fig1, ax1 = plt.subplots()
shap.summary_plot(shap_values, x_test)
st.pyplot(fig1)



def get_user_input():
    Branch_code = st.sidebar.selectbox('Branch code', np.sort(df['Branch_code'].unique()), key = '1')
    Vehicle_Make = st.sidebar.selectbox('Vehicle brand',np.sort(df['Vehicle_Make'].unique()), key = '2')
    Year_Manufacture = st.sidebar.selectbox('Recently Manufactured?', [1,0], key = '3')
    Loan_Tenure = st.sidebar.selectbox('Loan Tenure', np.sort(df['Loan_Tenure'].unique()), key = '4')
    Annual_Income = st.sidebar.slider('Annual_Income', np.sort(df['Annual_Income'].unique()), key = '5')
    Loan_Amount = st.sidebar.selectbox('Loan_Amount', np.sort(df['Loan_Amount'].unique()), key = '6'))


    user_data = {'Branch_code':Branch_code, 'Vehicle_Make' : Vehicle_Make, 'Year_Manufacture' :Year_Manufacture, 'Loan_Tenure' : Loan_Tenure,
                 'Annual_Income' : Annual_Income, 'Loan_Amount': Loan_Amount }

    features = pd.DataFrame(user_data, index = [0])

    return features

st.dataframe(get_user_input())