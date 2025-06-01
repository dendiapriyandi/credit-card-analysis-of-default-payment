import streamlit as st
import eda
import prediction

st.set_page_config(page_title='Credit Card Payment Default Prediction',
     layout='wide',
     initial_sidebar_state='expanded',
     page_icon=':credit_card:')

page = st.sidebar.selectbox('Select Page : ', ('Exploratory Data Analysis', 'Predict A Default Payment'))

if page == 'Exploratory Data Analysis':
    eda.run()
else:
    prediction.run()