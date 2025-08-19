# The code below was inspired by the Churnometer Project from Code Institute 
# with some adjustments
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from src.data_management import load_breast_cancer_data


def page_breast_cancer_diagnosis_body():
    # load data
    df = load_breast_cancer_data()

    # hard copied from breast cancer diagnosis study notebook
    vars_to_study = ['area_worst', 'concave points_mean', 'concave points_worst', 'perimeter_mean', 'perimeter_worst', 'radius_worst']

    st.write("### Breast Cancer Diagnosis Study")
    st.info(
        f"* The client is interested in understanding the patterns from the breast cancer dataset, "
        f"so that the client can learn the most relevant variables correlated "
        f"to breast cancer diagnosis.")

    # inspect data
    if st.checkbox("Inspect Breast Cancer Dataset"):
        st.write(
            f"* The dataset has {df.shape[0]} rows and {df.shape[1]} columns, "
            f"find below the first 10 rows.")

        st.write(df.head(10))

    st.write("---")

    # Correlation Study Summary
    st.write(
        f"* A correlation study was conducted in the jupyter notebook to better understand how "
        f"the variables are correlated to breast cancer diagnosis. \n"
        f"The most correlated variable are: **{vars_to_study}**"
    )

    # Text based on "02 - Cancer Diagnosis Study" notebook - "Conclusions and Next steps" section
    st.write(
        f"The correlation indications and plots cover the interpretations below -\n"
        f"* Higher worst area value might point to a Malignant diagnosis.\n"
        f"* Mean of the concave points if >0.05 might point to a Malignant diagnosis.\n"
        f"* Concave worst area value if >0.14 might point to a Malignant diagnosis.\n"
        f"* A mean tumor boundary(perimeter) value of >85 might point to a Malignant diagnosis.\n"
        f"* A >100 value of outer perimeter of lobes might point to a Malignant diagnosis.\n"
        f"* Higher worst radius value might point to a Malignant diagnosis."
    )

    # Code copied from "02 - Cancer Diagnosis Study" notebook - "EDA on selected variables" section
    df_eda = df.filter(vars_to_study + ['diagnosis'])

    # Individual plots per variable
    if st.checkbox("Diagnosis Levels per Variable"):
        diagnosis_level_per_variable(df_eda)

    # Multivariate plots
    if st.checkbox("Diagnosis Levels - Multivariate"):
        diagnosis_level_multivariate(df_eda)


def diagnosis_level_per_variable(df_eda):
    st.write(
        f"Visualize variable correlation to Diagnosis:"
    )
    target_var = 'diagnosis'
    for col in df_eda.columns:
        plot_numerical(df_eda, col, target_var)
        print("\n\n")


def plot_numerical(df_eda, col, target_var):
    fig, axes = plt.subplots(figsize=(8, 5))
    sns.histplot(data=df_eda, x=col, hue=target_var, kde=True, element="step", ax=axes)
    axes.set_title(f"{col}", fontsize=20, y=1.05)
    st.pyplot(fig)


def diagnosis_level_multivariate(df_eda):
    st.write(
        f"Multivariate analysis (MVA) is a set of statistical methods used to analyze "
        f"data sets with multiple variables, examining relationships and patterns among them. "
        f"We will visualize the MVA among the variables, all in one go, with a pairplot figure."
    )
    fig = sns.pairplot(df_eda, hue='diagnosis', corner=True, diag_kind='kde')
    plt.suptitle('Pairplot of Selected Variables', y=1.02, fontsize=20)
    st.pyplot(fig)
