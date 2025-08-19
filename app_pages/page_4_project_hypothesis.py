import streamlit as st


def page_project_hypothesis_body():
    st.write("### Project Hypothesis and Validation")

    st.success(
        f"Hypothesis 1: "
        f"***Breast tumors with larger worst area (>1,000 units) and higher perimeter (>85 units) are more likely to be Malignant.***\n"
        f"* The correlation study from the Breast Cancer Diagnosis Study page supports that, "
        f"tumor size and spread play a significant role in diagnosis. \n\n"
    )

    st.success(
        f"Hypothesis 2: "
        f"***The shape irregularity of the tumor, measured by concavity_mean (>0.05) and concavity_worst (>0.14), strongly indicates a Malignant diagnosis.***\n"
        f"* Multivariate analysis (MVA) study presented in the Breast Cancer Diagnosis Study page supports that, "
        f"boundary irregularity is a strong diagnostic signal.\n\n"
    )

    st.info(
        f"This information is crucial for guiding the diagnostic process and treatment planning for patients with breast cancer."
    )
