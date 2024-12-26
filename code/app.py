import os
import streamlit as st
import pandas as pd
import pickle
working_dir = os.getcwd()
#working_dir ='..'  # Use on Jupyter Notebook

#for running the app
# streamlit run ./code/app.py --server.port=8501 --server.address=127.0.0.1




st.write("""
# Feed Prediction App

This app predicts the **EAF Feeds** type!
""")

def coke1030_input_features():
    c_coke1030 = st.sidebar.slider('C (%)', 80.0, 85.0, 82.0)
    print(c_coke1030)
    s_coke1030 = st.sidebar.slider('S (%)', 0.0, 2.0, 1.0)
    s112_coke1030 = st.sidebar.slider('Size 112 (%)', 0.0, 5.0, 2.0)
    clf = pickle.load(open(f"{working_dir}/trained_models/coke_1030_rfc_model.pkl", 'rb'))
    data = {'c': c_coke1030,
            's': s_coke1030,
            's112': s112_coke1030,
            }
    features = pd.DataFrame(data, index=['Coke 1030'])
    features['Type prediction'] = clf.predict(features)
    return features




st.sidebar.header('Input Coke (1030) Parameters:')
coke_1030_df = coke1030_input_features()

st.subheader('User Input parameters')
st.write(coke_1030_df)
