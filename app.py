import streamlit as st
import pandas as pd
import joblib
from sklearn.svm import LinearSVC 

# Load model and scaler
model = joblib.load('linearsvc_retinopathy_model.joblib')
scaler = joblib.load('standard_scaler_retinopathy.joblib')

# Page Configuration
st.set_page_config(page_title="Retinopathy Detection System", page_icon="👁️")

# Title and Description
st.title("👁️ Retinopathy Prediction Interface")
st.markdown("""
This application uses a machine learning model to detect the presence of retinopathy. 
Please provide the clinical parameters below to obtain an automated prediction.
""")

# Form creation
with st.form("retinopathy_form"):
    st.subheader("Patient Clinical Data")
    col1, col2 = st.columns(2)
    
    with col1:
        pre_screening = st.number_input("Pre-screening Result", value=0)
        ma1 = st.number_input("Microaneurysms - MA1", value=38)
        ma2 = st.number_input("Microaneurysms - MA2", value=36)
        ma3 = st.number_input("Microaneurysms - MA3", value=35)
        ma4 = st.number_input("Microaneurysms - MA4", value=32)
        ma5 = st.number_input("Microaneurysms - MA5", value=28)
        ma6 = st.number_input("Microaneurysms - MA6", value=21)
        exudate1 = st.number_input("Exudates - E1", value=64.0)
        exudate2 = st.number_input("Exudates - E2", value=23.0)
        exudate3 = st.number_input("Exudates - E3", value=8.0)
        
    with col2:
        exudate4 = st.number_input("Exudates - E4", value=1.0)
        exudate5 = st.number_input("Exudates - E5", value=0.0)
        exudate6 = st.number_input("Exudates - E6", value=0.0)
        exudate7 = st.number_input("Exudates - E7", value=0.0)
        exudate8 = st.number_input("Exudates - E8", value=0.0)
        macula_opticdsc_distance = st.number_input("Macula-Optic Disc Distance", value=0.52)
        opticdisc_diameter = st.number_input("Optic Disc Diameter", value=0.10)
        am_fm_classification = st.number_input("AM/FM Classification", value=0)
        
    submit_button = st.form_submit_button("Run Prediction")

if submit_button:
    # Prepare data in the sequence expected by the model
    input_data = pd.DataFrame([[pre_screening, ma1, ma2, ma3, ma4, ma5, ma6, 
                              exudate1, exudate2, exudate3, exudate4, exudate5, exudate6, exudate7, exudate8, 
                              macula_opticdsc_distance, opticdisc_diameter, am_fm_classification]], 
                            columns=['pre_screening', 'ma1', 'ma2', 'ma3', 'ma4', 'ma5', 'ma6', 
                                     'exudate1', 'exudate2', 'exudate3', 'exudate4', 'exudate5', 'exudate6', 'exudate7', 'exudate8', 
                                     'macula_opticdsc_distance', 'opticdisc_diameter', 'am_fm_classification'])
    
    # Scale and Predict
    scaled_data = scaler.transform(input_data)
    prediction = model.predict(scaled_data)
    
    # Show Results
    st.divider()
    if prediction[0] == 1:
        st.error("### Result: Retinopathy Detected (Positive)")
        st.write("Clinical recommendation: Please consult with an ophthalmologist for further examination.")
    else:
        st.success("### Result: No Retinopathy Detected (Negative)")
        st.write("The analysis indicates no immediate signs of retinopathy based on provided data.")
