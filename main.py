import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
sns.set()
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import shap
import pickle
import base64
import streamlit.components.v1 as components
main_bg = "silver.png"
main_bg_ext = "png"

side_bg = "mustard.jpg"
side_bg_ext = "jpg"

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
st.write(""" ## Auto Car Loan Interest Rate Calculator""")



df = pd.read_csv('autoloan_super_new.csv')

x = df.drop("Interest_Rate", axis=1)
y = df["Interest_Rate"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

cb = pickle.load(open('catboost.sav', 'rb'))

st.sidebar.write(""" ### Interest Rate Calculator """)
def get_user_input():
    Branch_name = st.sidebar.selectbox('Branch_name', np.sort(df['Branch_name'].unique()), key='1')
    Vehicle_Make = st.sidebar.selectbox('Vehicle_Make', np.sort(df['Vehicle_Make'].unique()), key='2')
    Year_Manufacture = st.sidebar.selectbox('Year_Manufacture', [2020,2019,2018,2017,2016,2015], key='3')
    Loan_Tenure = st.sidebar.selectbox('Loan_Tenure', np.sort(df['Loan_Tenure'].unique())[::-1], key='4')
    Annual_Income = st.sidebar.selectbox('Annual_Income', np.sort(df['Annual_Income'].unique()), key='5')
    Loan_Amount = st.sidebar.number_input('Loan_Amount', 10000,2000000,70000, key='6')

    user_data = {'Branch_name': Branch_name, 'Vehicle_Make': Vehicle_Make, 'Year_Manufacture': Year_Manufacture,
                 'Loan_Tenure': Loan_Tenure,
                 'Annual_Income': Annual_Income, 'Loan_Amount': Loan_Amount}

    features = pd.DataFrame(user_data, index=[0])

    return features

user_input = get_user_input()
st.sidebar.write('Interest Rate Estimate:')
st.sidebar.write(cb.predict(user_input))

ex1 = shap.TreeExplainer(cb)
shap_values1 = ex1.shap_values(user_input)
shap.initjs()

def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)


if st.sidebar.button('Explain Prediction:'):
    st.write("Prediction Explaination:")
    st_shap(shap.force_plot(ex1.expected_value, shap_values1[0, :], user_input))

st.write(""" ### Datasets for Modelling """)
st.dataframe(df)

st.write(""" ### Distribution of the Datasets""")
def distribution_graph(column):
    fig, ax = plt.subplots(figsize = (12,8))
    if (column == 'Loan_Amount') or (column == 'Interest_Rate'):
        sns.distplot(df[column])
    elif (column == 'Annual_Income') or (column == 'Year_Manufacture'):
        df.groupby(column).agg({'Interest_Rate': 'count'}).sort_values(by='Interest_Rate', ascending=False).plot(kind='bar', legend=False, ax=ax, title='Total count of respective ' + column)
    else:
        k = st.slider('Top:', 1, df[column].nunique(), 10, key='10')
        df.groupby(column).agg({'Interest_Rate':'count'}).sort_values(by = 'Interest_Rate', ascending = False).iloc[:k].plot(kind='bar', legend=False, ax=ax, title = 'Total count of respective ' + column)
    st.pyplot(fig)

def plot_distribution():
    column =  st.selectbox('Feature:', ['Loan_Amount', 'Interest_Rate', 'Annual_Income', 'Year_Manufacture', 'Vehicle_Make',  'Branch_name'], key = '9')
    distribution_graph(column)

plot_distribution()

st.write(""" ### Estimator against Interest Rate Visualisation""")
def bar_graph(column):
    fig, ax = plt.subplots(figsize = (12,8))
    df.groupby(column).agg({'Interest_Rate': 'mean'}).sort_values(by = 'Interest_Rate', ascending = False).plot(kind='bar', ax = ax, title = 'Interest Rate (Mean) of each ' + column)
    st.pyplot(fig)

def bar_graph_slider(column, k):
    fig, ax = plt.subplots(figsize = (12,8))
    df.groupby(column).agg({'Interest_Rate': 'mean'}).sort_values(by = 'Interest_Rate', ascending = False).iloc[:k].plot(kind='bar', ax = ax, title = 'Interest Rate (Mean) of each ' + column)
    st.pyplot(fig)

def scatter_plot():
    fig, ax = plt.subplots()
    sns.scatterplot(x = 'Loan_Amount', y = 'Interest_Rate', data = df)
    st.pyplot(fig)

def plot_graph():
    column = st.selectbox('Feature:', ['Loan_Amount', 'Annual_Income', 'Year_Manufacture', 'Vehicle_Make',  'Branch_name'], key = '7')
    if column == 'Loan_Amount':
        scatter_plot()
    elif (column == 'Annual_Income') or (column == 'Year_Manufacture'):
        bar_graph(column)
    else:
        k = st.slider('Top:', 1,df[column].nunique(),  5, key = '8')
        bar_graph_slider(column, k)

plot_graph()

st.write(""" ### CatBoost Perfomance """)
df1 = pd.DataFrame(index=['R-Squared', 'Root Mean Squared Error'])
df1['Train Score'] = [cb.score(x_train,y_train), np.sqrt(mean_squared_error(y_train, cb.predict(x_train)))]
df1['Test Score'] = [cb.score(x_test,y_test), np.sqrt(mean_squared_error(y_test, cb.predict(x_test)))]
df1

st.write(""" ### Feature Importance of the Estimators""")
ex = shap.TreeExplainer(cb)
shap_values = ex.shap_values(x_test)
shap.initjs()

fig, ax = plt.subplots()
shap.summary_plot(shap_values, x_test, plot_type = 'bar')
st.pyplot(fig)

st.write(""" ### Shap Values""")
fig1, ax1 = plt.subplots()
shap.summary_plot(shap_values, x_test)
st.pyplot(fig1)

