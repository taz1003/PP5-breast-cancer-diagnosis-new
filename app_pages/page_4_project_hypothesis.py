import streamlit as st


def page_project_hypothesis_body():
    st.write("### Project Hypothesis and Validation")

    st.success(
        f"Hypothesis 1: "
        f"***Higher tumor size and area (area_mean, area_worst) significantly increase the likelihood of malignant diagnosis.***\n"
        f"* Validated via correlation analysis and feature importance from AdaBoost.\n"
        f"* The correlation study using Spearman correlation method visualized in the Breast Cancer Diagnosis Study page supports that.\n\n"
    )

    st.success(
        f"Hypothesis 2: "
        f"***Higher concavity and perimeter features are strongly linked with malignant diagnosis.***\n"
        f"Validated via clustering and threshold-based plots.\n"
        f"* Multivariate analysis (MVA) study presented in the Breast Cancer Diagnosis Study page supports that.\n\n"
    )

    st.info(
        f"This information is crucial for guiding the diagnostic process and treatment planning for patients with breast cancer."
    )
