import streamlit as st
import pandas as pd
from prediction import predict

st.title("Predicting Thyroid Cancer Recurrence")
st.markdown('Bayesian Classifier Machine Learning model trained to predict Thyroid Cancer Recurrence.')

st.header('Features')

age = st.number_input("Age", 0, 110)
gender = st.selectbox("Gender", ["F", "M"])
smoking = st.selectbox("Smoking", ['Yes', 'No'])
hx_Smoking = st.selectbox("History of Smoking", ['Yes', 'No'])
hx_Radiothreapy = st.selectbox("History of Radiotherapy", ['Yes', 'No'])
thyrdoi_Function = st.selectbox("Thyroid Function", ['Euthyroid', 'Clinical Hyperthyroidism', 'Clinical Hypothyroidism', 'Subclinical Hyperthyroidism', 'Subclinical Hypothyroidism'])
physical_Exam = st.selectbox("Physical Exam", ['Normal', 'Single nodular goiter-left', 'Single nodular goiter-right', 'Multinodular goiter', 'Diffuse goiter'])
adenopathy = st.selectbox("Adenopathy", ['No', 'Right', 'Left', 'Extensive', 'Bilateral', 'Posterior'])
pathology = st.selectbox("Pathology", ['Micropapillary', 'Papillary', 'Follicular', 'Hurthel cell'])
focality = st.selectbox("Focality", ['Uni-Focal', 'Multi-Focal'])
risk = st.selectbox("Risk", ['Low', 'Intermediate', 'High'])
T = st.selectbox("Tumour Size", ['T1a', 'T1b', 'T2', 'T3a', 'T3b', 'T4a', 'T4b'])
M = st.selectbox("Metastatsis", ['M0', 'M1'])
N = st.selectbox("Lymph Nodes", ['N0', 'N1a', 'N1b'])
stage = st.selectbox("Stage", ['I', 'II', 'III','IVA','IVB'])
response = st.selectbox("Response", ['Indeterminate', 'Excellent', 'Structural Incomplete', 'Biochemical Incomplete'])

if st.button('Predict Thyroid Cancer Recurrence'):
    values = [age, gender, smoking, hx_Smoking, hx_Radiothreapy, thyrdoi_Function, physical_Exam, adenopathy, pathology, focality, risk, T, N, M, stage, response]
    columns = ['Age','Gender', 'Smoking', 'Hx Smoking', 'Hx Radiothreapy', 'Thyroid Function', 'Physical Examination', 'Adenopathy', 'Pathology', 'Focality', 'Risk', 'T', 'N', 'M', 'Stage', 'Response']
    input_data = pd.DataFrame([values], columns=columns)
    print(type(input_data))
    result = predict(input_data)
    st.text(result[0])