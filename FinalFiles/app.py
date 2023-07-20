#!/usr/bin/env python
# coding: utf-8

# # Building the Streamlit app

# In[6]:


import streamlit as st 
import pandas as pd
import numpy as np
from prediction import predict


# In[7]:


st.title('Classifying Iris Flowers')
st.markdown('Toy model to play to classify iris flowers into \
setosa, versicolor, virginica')


# In[8]:


st.header('Plant Features')
col1, col2 = st.columns(2)
with col1:
    st.text('Sepal characteristics')
    sepal_l = st.slider('Sepal lenght (cm)', 1.0, 8.0, 0.5)
    sepal_w = st.slider('Sepal width (cm)', 2.0, 4.4, 0.5)
with col2:
    st.text('Pepal characteristics')
    petal_l = st.slider('Petal lenght (cm)', 1.0, 7.0, 0.5)
    petal_w = st.slider('Petal width (cm)', 0.1, 2.5, 0.5)


# In[9]:


st.button('Predict type of Iris')


# In[10]:


if st.button('Predict type of Iris'):
    result = predict(np.array([[sepal_l, sepal_w, petal_l, petal_w]]))
    st.text(result[0])


# In[ ]:





# In[ ]:





# In[ ]:




