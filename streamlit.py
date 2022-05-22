import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np


import os
import requests


#selects first 100 rows
df_app = pd.read_csv('df_app.csv')
users=df_app['user_id'][:100].unique()

#opens bookshelf image
image = Image.open('icon.png')
st.image(image,use_column_width=True)
st.write("""## Welcome to Bookshelf""")

#box that is used for app user to select a userid
userId = st.selectbox('Please select a userID', options=users)

#url = 'https://p9deployapp1.azurewebsites.net/api/httptrigger10'
url = 'http://localhost:7071/api/HttpTrigger10'
if st.button('Connect'):
    myobj = {'name': userId}
    x = requests.get(url, params = myobj)
    st.write("##  We recommend the following articles : ")
    st.write(x)
    st.write(x.text)
   
   
  
   



