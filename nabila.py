import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('2018-2010_export.csv')
df.head()

st.dataframe(df)

st.write('Hi Nabila')

fig, ax = plt.subplots()
df.groupby('year').agg({'value':'sum'}).plot(ax=ax)
st.pyplot(fig)

