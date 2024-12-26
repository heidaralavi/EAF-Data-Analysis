import os
import streamlit as st
import pandas as pd
import pickle
#working_dir = os.getcwd()
working_dir ='..'  # Use on Jupyter Notebook

#for running the app
# streamlit run ./code/app.py --server.port=8501 --server.address=127.0.0.1




st.write("""
# Feed Prediction App

This app predicts the **EAF Feeds** type!
""")

def coke1030_input_features():
    c_coke1030 = st.sidebar.slider('C (%)', 80.0, 85.0, 82.0,key='coke1030_1')
    s_coke1030 = st.sidebar.slider('S (%)', 0.0, 2.0, 1.0,key='coke1030_2')
    s112_coke1030 = st.sidebar.slider('Size 112 (%)', 0.0, 5.0, 2.0,key='coke1030_3')
    clf = pickle.load(open(f"./trained_models/coke_1030_rfc_model.pkl", 'rb'))
    data = {'c': c_coke1030,
            's': s_coke1030,
            's112': s112_coke1030,
            }
    features = pd.DataFrame(data, index=['p'])
    features['Coke 1030 Type'] = clf.predict(features)
    return features

def cokefine_input_features():
    c_cokefine = st.sidebar.slider('C (%)', 70.0, 100.0, 85.0,key='cokefine_1')
    s_cokefine = st.sidebar.slider('S (%)', 0.0, 2.0, 1.0,key='cokefine_2')
    s05_cokefine = st.sidebar.slider('Size 05 (%)', 0.0, 5.5, 2.0,key='cokefine_3')
    clf = pickle.load(open(f"./trained_models/coke_fine_rfc_model.pkl", 'rb'))
    data = {'c': c_cokefine,
            's': s_cokefine,
            's05': s05_cokefine,
            }
    features = pd.DataFrame(data, index=['p'])
    features['Coke Fine Type'] = clf.predict(features)
    return features



st.sidebar.header('Input Coke (1030) Parameters:')
coke_1030_df = coke1030_input_features()
st.sidebar.header('Input Coke Fine Parameters:')
coke_fine_df = cokefine_input_features()

st.subheader('User Input parameters')
st.write(pd.concat([coke_1030_df['Coke 1030 Type'],coke_fine_df['Coke Fine Type']],axis=1))
