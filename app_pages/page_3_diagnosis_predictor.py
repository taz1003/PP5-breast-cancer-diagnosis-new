import streamlit as st
import pandas as pd
from src.data_management import load_breast_cancer_data, load_pkl_file
from src.machine_learning.predictive_analysis_ui import (
    predict_diagnosis,
    predict_cluster)


def page_patient_body():

    # load predict diagnosis files
    version = 'v1'
    diagnosis_pipe_dc_fe = load_pkl_file(
        f'outputs/ml_pipeline/predict_diagnosis/{version}/clf_pipeline_data_cleaning_feat_eng.pkl')
    diagnosis_pipe_model = load_pkl_file(
        f"outputs/ml_pipeline/predict_diagnosis/{version}/clf_pipeline_model.pkl")
    diagnosis_features = (pd.read_csv(f"outputs/ml_pipeline/predict_diagnosis/{version}/X_train.csv")
                          .columns
                          .to_list()
                          )

    # load cluster analysis files
    version = 'v1'
    cluster_pipe = load_pkl_file(
        f"outputs/ml_pipeline/cluster_analysis/{version}/cluster_pipeline.pkl")
    cluster_features = (pd.read_csv(f"outputs/ml_pipeline/cluster_analysis/{version}/TrainSet.csv")
                        .columns
                        .to_list()
                        )
    cluster_profile = pd.read_csv(
        f"outputs/ml_pipeline/cluster_analysis/{version}/clusters_profile.csv")

    st.write("### Diagnosis Predictor Interface")
    st.info(
        f"* The client is interested in determining whether or not a patient has breast cancer. "
        f"In addition, the client is "
        f"interested in learning from which cluster this patient will belong in the diagnosis base.\n\n"
        f"* Based on that, the medical team will present potential treatment options to the patient."
    )
    statement_warning = (
        f"* It is to be noted that cluster profiles use only four top features for"
        f" interpretation, which may differ from the overall diagnosis probability.\n\n"
        f"* It is still suggested to take cluster profiles into account earnestly "
        f"since they can provide valuable insights into the patient's condition."
    )
    st.warning(statement_warning)
    st.write("---")

    st.write("### Predictive Analysis UI with the Most Important Features")
    st.write(
        f"* The UI allows you to input the most important features for the diagnosis and cluster analysis."
    )
    # Generate Live Data
    # check_variables_for_UI(diagnosis_features, cluster_features)
    X_live = DrawInputsWidgets()

    # predict on live data
    if st.button("Run Predictive Analysis"):
        diagnosis_prediction = predict_diagnosis(
            X_live, diagnosis_features, diagnosis_pipe_dc_fe, diagnosis_pipe_model)

        predict_cluster(X_live, cluster_features,
                        cluster_pipe, cluster_profile)


def check_variables_for_UI(diagnosis_features, cluster_features):
    import itertools

    # The widgets inputs are the features used in all pipelines (diagnosis, cluster)
    # We combine them only with unique values
    combined_features = set(
        list(
            itertools.chain(diagnosis_features, cluster_features)
        )
    )
    st.write(
        f"* There are {len(combined_features)} features for the UI: \n\n {combined_features}")


def DrawInputsWidgets():

    # load dataset
    df = load_breast_cancer_data()
    percentageMin, percentageMax = 0.4, 2.0

    # create input widgets for 10 features
    col1, col2, col3 = st.columns(3)
    col4, col5, col6 = st.columns(3)
    col7, col8, col9 = st.columns(3)

    # empty DataFrame for live data
    X_live = pd.DataFrame([], index=[0])

    # number input for continuous variables

    def add_number_input(col, feature):
        with col:
            st_widget = st.number_input(
                label=feature,
                min_value=float(df[feature].min() * percentageMin),
                max_value=float(df[feature].max() * percentageMax),
                value=float(df[feature].median())
            )
        X_live[feature] = st_widget

    # Now add your 10 features
    add_number_input(col1, "area_mean")
    add_number_input(col2, "smoothness_worst")
    add_number_input(col3, "perimeter_se")
    add_number_input(col4, "texture_worst")
    add_number_input(col5, "fractal_dimension_se")
    add_number_input(col6, "concavity_mean")
    add_number_input(col7, "perimeter_worst")
    add_number_input(col8, "fractal_dimension_worst")
    add_number_input(col9, "perimeter_mean")

    # return the live dataframe
    return X_live