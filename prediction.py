#!/usr/bin/env python
# coding: utf-8

# # Prediction engine for streamlit

# In[3]:


import joblib
def predict(data):
    clf = joblib.load('rf_model.sav')
    return clf.predict(data)


# In[ ]:




