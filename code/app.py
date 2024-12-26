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
    c_coke1030 = st.sidebar.slider('C (%)', 80.0, 85.0, 82.0,key='coke1030_1')
    s_coke1030 = st.sidebar.slider('S (%)', 0.0, 2.0, 1.0,key='coke1030_2')
    s112_coke1030 = st.sidebar.slider('Size 112 (%)', 0.0, 5.0, 2.0,key='coke1030_3')
    clf = pickle.load(open(f"{working_dir}/trained_models/coke_1030_rfc_model.pkl", 'rb'))
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
    clf = pickle.load(open(f"{working_dir}/trained_models/coke_fine_rfc_model.pkl", 'rb'))
    data = {'c': c_cokefine,
            's': s_cokefine,
            's05': s05_cokefine,
            }
    features = pd.DataFrame(data, index=['p'])
    features['Coke Fine Type'] = clf.predict(features)
    return features

def dolo_input_features():
    cao_dolo = st.sidebar.slider('Cao (%)', 45.0, 60.0, 55.0,key='dolo_1')
    mgo_dolo = st.sidebar.slider('Mgo (%)', 30.0, 40.0, 33.0,key='dolo_2')
    s0_95_dolo = st.sidebar.slider('Size 0-95 (%)', 0.0, 10.0, 2.0,key='dolo_3')
    s95_385_dolo = st.sidebar.slider('Size 95-385 (%)', 0.0, 100.0, 55.0,key='dolo_4')
    s385_1000_dolo = st.sidebar.slider('Size 385-1000 (%)', 0.0, 20.0, 2.0,key='dolo_5')
    clf = pickle.load(open(f"{working_dir}/trained_models/dolomite_rfc_model.pkl", 'rb'))
    data = {'cao': cao_dolo,
            'mgo': mgo_dolo,
            's095': s0_95_dolo,
            's95_385': s95_385_dolo,
            's385_1000': s385_1000_dolo,
            }
    features = pd.DataFrame(data, index=['p'])
    features['Dolomite Type'] = clf.predict(features)
    return features


def dri_input_features():
    fe_metal_dri = st.sidebar.slider('Fe_Metal (%)', 70.0, 85.0, 80.0,key='dri_1')
    fe_total_dri = st.sidebar.slider('Fe_total (%)', 80.0, 90.0, 84.0,key='dri_2')
    md_dri = st.sidebar.slider('MD (%)', 85.0, 100.0, 90.0,key='dri_3')
    c_dri = st.sidebar.slider('C (%)', 0.0, 3.0, 1.2,key='dri_4')
    cao_dri = st.sidebar.slider('Cao (%)', 0.0, 1.0, 1.0,key='dri_5')
    sio2_dri = st.sidebar.slider('Sio2 (%)', 0.0, 6.0, 2.0,key='dri_6')
    mgo_dri = st.sidebar.slider('Mgo (%)', 0.0, 1.5, 0.8,key='dri_7')
    al2o3_dri = st.sidebar.slider('Al2o3 (%)', 0.0, 1.5, 0.8,key='dri_8')
    p_dri = st.sidebar.slider('P (%)', 0.0, 0.15, 0.08,key='dri_9')
    mno_dri = st.sidebar.slider('Mno (%)', 0.0, 0.05, 0.02,key='dri_10')
    gunge_dri = st.sidebar.slider('Gunge (%)', 7.0, 15.0, 9.5,key='dri_11')
    feo_dri = st.sidebar.slider('Feo (%)', 6.0, 15.0, 9.5,key='dri_12')
    feo_c_dri = st.sidebar.slider('Feo_c (%)', 0.0, 12.0, 9.5,key='dri_13')
    clf = pickle.load(open(f"{working_dir}/trained_models/dri_rfc_model.pkl", 'rb'))
    data = {'fe_metal': fe_metal_dri,
            'fe_total': fe_total_dri,
            'md': md_dri,
            'c': c_dri,
            'cao': cao_dri,
            'sio2': sio2_dri,
            'mgo': mgo_dri,
            'al2o3': al2o3_dri,
            'p': p_dri,
            'mno': mno_dri,
            'gunge': gunge_dri,
            'feo': feo_dri,
            'feo_c': feo_c_dri,
                        }
    features = pd.DataFrame(data, index=['p'])
    features['Dri Type'] = clf.predict(features)
    return features




st.sidebar.header('Input Coke (1030) Parameters:')
coke_1030_df = coke1030_input_features()
st.sidebar.header('Input Coke Fine Parameters:')
coke_fine_df = cokefine_input_features()
st.sidebar.header('Input Dolomite Parameters:')
dolo_df = dolo_input_features()
st.sidebar.header('Input DRI Parameters:')
dri_df = dri_input_features()

st.subheader('User Input parameters')
st.write(pd.concat([coke_1030_df['Coke 1030 Type'],coke_fine_df['Coke Fine Type'],dolo_df['Dolomite Type']
                    ,dri_df['Dri Type']],axis=1))
