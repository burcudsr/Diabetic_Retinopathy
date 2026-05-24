import streamlit as st
import pandas as pd
import joblib
from sklearn.svm import LinearSVC # LinearSVC'yi açıkça import ediyoruz

# Modelleri yükle
model = joblib.load('linearsvc_retinopathy_model.joblib')
scaler = joblib.load('standard_scaler_retinopathy.joblib')

st.set_page_config(page_title="Retinopathy Prediction", page_icon="👁️")
st.title("👁️ Retinopathy Prediction Interface")
st.write("Please enter the patient's data below.")

with st.form("retinopathy_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        pre_screening = st.number_input("Pre-screening", value=0)
        ma1 = st.number_input("Microaneurysms (ma1)", value=38)
        ma2 = st.number_input("Microaneurysms (ma2)", value=36)
        ma3 = st.number_input("Microaneurysms (ma3)", value=35)
        ma4 = st.number_input("Microaneurysms (ma4)", value=32)
        ma5 = st.number_input("Microaneurysms (ma5)", value=28)
        ma6 = st.number_input("Microaneurysms (ma6)", value=21)
        exudate1 = st.number_input("Exudates 1", value=64.0)
        exudate2 = st.number_input("Exudates 2", value=23.0)
        exudate3 = st.number_input("Exudates 3", value=8.0)
        
    with col2:
        exudate4 = st.number_input("Exudates 4", value=1.0)
        exudate5 = st.number_input("Exudates 5", value=0.0)
        exudate6 = st.number_input("Exudates 6", value=0.0)
        exudate7 = st.number_input("Exudates 7", value=0.0)
        exudate8 = st.number_input("Exudates 8", value=0.0)
        macula_opticdsc_distance = st.number_input("Macula Optic Disc Distance", value=0.52)
        opticdisc_diameter = st.number_input("Optic Disc Diameter", value=0.10)
        am_fm_classification = st.number_input("AM/FM Classification", value=0)
        
    submit_button = st.form_submit_button("Predict")

if submit_button:
    # Veriyi modelin beklediği sırayla DataFrame yap
    features = pd.DataFrame([[pre_screening, ma1, ma2, ma3, ma4, ma5, ma6, 
                              exudate1, exudate2, exudate3, exudate4, exudate5, exudate6, exudate7, exudate8, 
                              macula_opticdsc_distance, opticdisc_diameter, am_fm_classification]], 
                            columns=['pre_screening', 'ma1', 'ma2', 'ma3', 'ma4', 'ma5', 'ma6', 
                                     'exudate1', 'exudate2', 'exudate3', 'exudate4', 'exudate5', 'exudate6', 'exudate7', 'exudate8', 
                                     'macula_opticdsc_distance', 'opticdisc_diameter', 'am_fm_classification'])
    
    # Ölçekle ve Tahmin Et
    scaled_data = scaler.transform(features)
    prediction = model.predict(scaled_data)
    
    if prediction[0] == 1:
        st.error("Prediction: Retinopathy Detected (Positive)")
    else:
        st.success("Prediction: No Retinopathy Detected (Negative)")
