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
**Desgin By heidar alavi @CMIC 2025**
# This app predicts the **EAF Feeds** type!
""")

def coke1030_input_features(min_dict,max_dict,avg_dict):
    c_coke1030 = st.sidebar.slider('C (%)', min_dict['c (Coke1030)'], max_dict['c (Coke1030)'], avg_dict['c (Coke1030)'],key='coke1030_1')
    s_coke1030 = st.sidebar.slider('S (%)', min_dict['s (Coke1030)'], max_dict['s (Coke1030)'], avg_dict['s (Coke1030)'],key='coke1030_2')
    s112_coke1030 = st.sidebar.slider('Size 112 (%)', min_dict['s112 (Coke1030)'], max_dict['s112 (Coke1030)'], avg_dict['s112 (Coke1030)'],key='coke1030_3')
    #clf = pickle.load(open(f"{working_dir}/trained_models/coke_1030_rfc_model.pkl", 'rb'))
    data = {'c_coke1030': c_coke1030,
            's_coke1030': s_coke1030,
            's112_coke1030': s112_coke1030,
            }
    features = pd.DataFrame(data, index=['p'])
    #features['Coke 1030 Type'] = clf.predict(features)
    return features

def cokefine_input_features(min_dict,max_dict,avg_dict):
    c_cokefine = st.sidebar.slider('C (%)', min_dict['c (CokeFine)'], max_dict['c (CokeFine)'], avg_dict['c (CokeFine)'],key='cokefine_1')
    s_cokefine = st.sidebar.slider('S (%)', min_dict['s (CokeFine)'], max_dict['s (CokeFine)'], avg_dict['s (CokeFine)'],key='cokefine_2')
    s05_cokefine = st.sidebar.slider('Size 05 (%)', min_dict['s05 (CokeFine)'], max_dict['s05 (CokeFine)'], avg_dict['s05 (CokeFine)'],key='cokefine_3')
    #clf = pickle.load(open(f"{working_dir}/trained_models/coke_fine_rfc_model.pkl", 'rb'))
    data = {'c_cokefine': c_cokefine,
            's_cokefine': s_cokefine,
            's05_cokefine': s05_cokefine,
            }
    features = pd.DataFrame(data, index=['p'])
    #features['Coke Fine Type'] = clf.predict(features)
    return features

def dolo_input_features(min_dict,max_dict,avg_dict):
    cao_dolo = st.sidebar.slider('Cao (%)', min_dict['cao (Dolomite)'], max_dict['cao (Dolomite)'],avg_dict['cao (Dolomite)'],key='dolo_1')
    mgo_dolo = st.sidebar.slider('Mgo (%)', min_dict['mgo (Dolomite)'], max_dict['mgo (Dolomite)'], avg_dict['mgo (Dolomite)'],key='dolo_2')
    s0_95_dolo = st.sidebar.slider('Size 0-95 (%)', min_dict['s0_95 (Dolomite)'], max_dict['s0_95 (Dolomite)'], avg_dict['s0_95 (Dolomite)'],key='dolo_3')
    #clf = pickle.load(open(f"{working_dir}/trained_models/dolomite_rfc_model.pkl", 'rb'))
    data = {'cao_dolo': cao_dolo,
            'mgo_dolo': mgo_dolo,
            's095_dolo': s0_95_dolo,
            }
    features = pd.DataFrame(data, index=['p'])
    #features['Dolomite Type'] = clf.predict(features)
    return features


def dri_input_features(min_dict,max_dict,avg_dict):
    fe_metal_dri = st.sidebar.slider('Fe_Metal (%)', min_dict['fe_metal (DRI)'], max_dict['fe_metal (DRI)'], avg_dict['fe_metal (DRI)'],key='dri_1')
    fe_total_dri = st.sidebar.slider('Fe_total (%)', min_dict['fe_total (DRI)'], max_dict['fe_total (DRI)'], avg_dict['fe_total (DRI)'],key='dri_2')
    md_dri = st.sidebar.slider('MD (%)', min_dict['md (DRI)'], max_dict['md (DRI)'], avg_dict['md (DRI)'],key='dri_3')
    c_dri = st.sidebar.slider('C (%)', min_dict['c (DRI)'], max_dict['c (DRI)'], avg_dict['c (DRI)'],key='dri_4')
    gunge_dri = st.sidebar.slider('Gunge (%)', min_dict['gunge (DRI)'], max_dict['gunge (DRI)'], avg_dict['gunge (DRI)'],key='dri_11')
    feo_dri = st.sidebar.slider('Feo (%)', min_dict['feo (DRI)'], max_dict['feo (DRI)'], avg_dict['feo (DRI)'],key='dri_12')
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

def lime_input_features(min_dict,max_dict,avg_dict):
    cao_lime = st.sidebar.slider('Cao (%)', min_dict['cao (Lime)'], max_dict['cao (Lime)'], avg_dict['cao (Lime)'],key='lime_1')
    mgo_lime = st.sidebar.slider('Mgo (%)', min_dict['mgo (Lime)'], max_dict['mgo (Lime)'], avg_dict['mgo (Lime)'],key='lime_2')
    s0_95_lime = st.sidebar.slider('Size 0-95 (%)', 0.0, 10.0, 2.0,key='lime_3')
    #clf = pickle.load(open(f"{working_dir}/trained_models/dolomite_rfc_model.pkl", 'rb'))
    data = {'cao_lime': cao_lime,
            'mgo_lime': mgo_lime,
            's095_lime': s0_95_lime,
            }
    features = pd.DataFrame(data, index=['p'])
    #features['Dolomite Type'] = clf.predict(features)
    return features

#---------------load data-------------------
df = pd.read_csv(f"{working_dir}/data/all_data_with_labels.csv")
max_dict = df.max().round(2).to_dict()
min_dict = df.min().round(2).to_dict()
avg_dict = df.mean().round(2).to_dict()

#---------------sidebar-------------------
st.sidebar.header('Input Coke (1030) Parameters:')
coke_1030_df = coke1030_input_features(min_dict=min_dict,max_dict=max_dict,avg_dict=avg_dict)
st.sidebar.header('Input Coke Fine Parameters:')
coke_fine_df = cokefine_input_features(min_dict=min_dict,max_dict=max_dict,avg_dict=avg_dict)
st.sidebar.header('Input Dolomite Parameters:')
dolo_df = dolo_input_features(min_dict=min_dict,max_dict=max_dict,avg_dict=avg_dict)
st.sidebar.header('Input Cold DRI Parameters:')
dri_df = dri_input_features(min_dict=min_dict,max_dict=max_dict,avg_dict=avg_dict)
st.sidebar.header('Input Lime Parameters:')
lime_df = lime_input_features(min_dict=min_dict,max_dict=max_dict,avg_dict=avg_dict)

#---------------main-------------------
st.subheader('User Input parameters')
prediction_df = pd.concat([coke_1030_df,coke_fine_df,dolo_df,dri_df,lime_df],axis=1)
st.write(prediction_df)

#--------load trained model and predict-----------
clf = pickle.load(open(f"{working_dir}/trained_models/all_feed_rfc_model.pkl", 'rb'))
predict_label = clf.predict(prediction_df.values).tolist()[0]
st.write(f"Predicted Label: {predict_label}")

#--------show the data for the predicted label-----------
mask = df.columns.str.contains('Coke1030|CokeFine|Dolomite|DRI|Lime')
df = df[df.columns[~mask]]
mask2 = df['Total_labels'] == predict_label
df = df[mask2]
mask3 = df['Slag_labels'] == 0
df = df[mask3]
max_b2 = df['b2 (Slag)'].max()
mask4 = df['b2 (Slag)'].between(0.5*max_b2,max_b2)
df = df[mask4]
min_feo = df['feo (Slag)'].min()
mask5 = df['feo (Slag)'].between(0.5*min_feo,min_feo)
df = df[mask5]
mask6 = df.columns.str.contains('EAF|Heat|feo|mgo|b2|Total_labels')
df = df[df.columns[mask6]]
st.write(df)
isd_df = pd.read_csv(f"{working_dir}/opencv-images/1xy.csv")
st.image(f"{working_dir}/opencv-images/isd-feo-mgo.jpg")
#st.scatter_chart(data=isd_df[isd_df['label']==0],x='x',y='y',x_label='Feo',y_label='Mgo')
