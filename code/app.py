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
    #clf = pickle.load(open(f"{working_dir}/trained_models/coke_1030_rfc_model.pkl", 'rb'))
    data = {'c_coke1030': c_coke1030,
            's_coke1030': s_coke1030,
            's112_coke1030': s112_coke1030,
            }
    features = pd.DataFrame(data, index=['p'])
    #features['Coke 1030 Type'] = clf.predict(features)
    return features

def cokefine_input_features():
    c_cokefine = st.sidebar.slider('C (%)', 70.0, 100.0, 85.0,key='cokefine_1')
    s_cokefine = st.sidebar.slider('S (%)', 0.0, 2.0, 1.0,key='cokefine_2')
    s05_cokefine = st.sidebar.slider('Size 05 (%)', 0.0, 5.5, 2.0,key='cokefine_3')
    #clf = pickle.load(open(f"{working_dir}/trained_models/coke_fine_rfc_model.pkl", 'rb'))
    data = {'c_cokefine': c_cokefine,
            's_cokefine': s_cokefine,
            's05_cokefine': s05_cokefine,
            }
    features = pd.DataFrame(data, index=['p'])
    #features['Coke Fine Type'] = clf.predict(features)
    return features

def dolo_input_features():
    cao_dolo = st.sidebar.slider('Cao (%)', 45.0, 60.0, 55.0,key='dolo_1')
    mgo_dolo = st.sidebar.slider('Mgo (%)', 30.0, 40.0, 33.0,key='dolo_2')
    s0_95_dolo = st.sidebar.slider('Size 0-95 (%)', 0.0, 10.0, 2.0,key='dolo_3')
    #clf = pickle.load(open(f"{working_dir}/trained_models/dolomite_rfc_model.pkl", 'rb'))
    data = {'cao_dolo': cao_dolo,
            'mgo_dolo': mgo_dolo,
            's095_dolo': s0_95_dolo,
            }
    features = pd.DataFrame(data, index=['p'])
    #features['Dolomite Type'] = clf.predict(features)
    return features


def dri_input_features():
    fe_metal_dri = st.sidebar.slider('Fe_Metal (%)', 70.0, 85.0, 80.0,key='dri_1')
    fe_total_dri = st.sidebar.slider('Fe_total (%)', 80.0, 90.0, 84.0,key='dri_2')
    md_dri = st.sidebar.slider('MD (%)', 85.0, 100.0, 90.0,key='dri_3')
    c_dri = st.sidebar.slider('C (%)', 0.0, 3.0, 1.2,key='dri_4')
    gunge_dri = st.sidebar.slider('Gunge (%)', 7.0, 15.0, 9.5,key='dri_11')
    feo_dri = st.sidebar.slider('Feo (%)', 6.0, 15.0, 9.5,key='dri_12')
    #clf = pickle.load(open(f"{working_dir}/trained_models/dri_rfc_model.pkl", 'rb'))
    data = {'fe_metal': fe_metal_dri,
            'fe_total': fe_total_dri,
            'md': md_dri,
            'c': c_dri,
            'gunge': gunge_dri,
            'feo': feo_dri,
            }
    features = pd.DataFrame(data, index=['p'])
    #features['Dri Type'] = clf.predict(features)
    return features

def lime_input_features():
    cao_lime = st.sidebar.slider('Cao (%)', 45.0, 60.0, 55.0,key='lime_1')
    mgo_lime = st.sidebar.slider('Mgo (%)', 30.0, 40.0, 33.0,key='lime_2')
    s0_95_lime = st.sidebar.slider('Size 0-95 (%)', 0.0, 10.0, 2.0,key='lime_3')
    #clf = pickle.load(open(f"{working_dir}/trained_models/dolomite_rfc_model.pkl", 'rb'))
    data = {'cao_lime': cao_lime,
            'mgo_lime': mgo_lime,
            's095_lime': s0_95_lime,
            }
    features = pd.DataFrame(data, index=['p'])
    #features['Dolomite Type'] = clf.predict(features)
    return features


st.sidebar.header('Input Coke (1030) Parameters:')
coke_1030_df = coke1030_input_features()
st.sidebar.header('Input Coke Fine Parameters:')
coke_fine_df = cokefine_input_features()
st.sidebar.header('Input Dolomite Parameters:')
dolo_df = dolo_input_features()
st.sidebar.header('Input DRI Parameters:')
dri_df = dri_input_features()
st.sidebar.header('Input Lime Parameters:')
lime_df = lime_input_features()


st.subheader('User Input parameters')
prediction_df = pd.concat([coke_1030_df,coke_fine_df,dolo_df,dri_df,lime_df],axis=1)
st.write(prediction_df)

clf = pickle.load(open(f"{working_dir}/trained_models/all_feed_rfc_model.pkl", 'rb'))
predict_label = clf.predict(prediction_df.values).tolist()[0]
st.write(f"Predicted Label: {predict_label}")
df = pd.read_csv(f"{working_dir}/data/all_data_with_labels.csv")
mask = df.columns.str.contains('Coke1030|CokeFine|Dolomite|DRI|Lime')
df = df[df.columns[~mask]]
mask2 = df['Total_labels'] == predict_label
df = df[mask2]
max_b2 = df['b2 (Slag)'].max()
mask3 = df['b2 (Slag)'] == max_b2
df = df[mask3]
mask4 = df.columns.str.contains('EAF|Heat|b2|Total_labels')
df = df[df.columns[mask4]]
st.write(df)
