import streamlit as st
from app_pages.multipage import MultiPage

# load pages scripts
from app_pages.page_1_summary import page_summary_body
from app_pages.page_2_breast_cancer_diagnosis_study import page_breast_cancer_diagnosis_body
from app_pages.page_3_diagnosis_predictor import page_patient_body
from app_pages.page_4_project_hypothesis import page_project_hypothesis_body
from app_pages.page_5_predict_diagnosis import page_predict_diagnosis_body
from app_pages.page_6_cluster import page_cluster_body

# Create an instance of the app
app = MultiPage(app_name="Breast Cancer Diagnosis")

# Adding app pages here using .add_page()
app.add_page("Quick Project Summary", page_summary_body)
app.add_page("Breast Cancer Diagnosis Study", page_breast_cancer_diagnosis_body)
app.add_page("Diagnosis Predictor - Breast Cancer", page_patient_body)
app.add_page("Project Hypothesis and Validation", page_project_hypothesis_body)
app.add_page("ML: Predict Breast Cancer Diagnosis", page_predict_diagnosis_body)
app.add_page("ML: Cluster Analysis", page_cluster_body)

app.run()
