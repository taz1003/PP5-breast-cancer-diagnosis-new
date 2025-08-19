# The code below was inspired by the Churnometer Project from Code Institute 
# with some adjustments
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from src.data_management import load_pkl_file
from src.machine_learning.evaluate_clf import clf_performance


def page_predict_diagnosis_body():

    version = 'v1'
    # load data
    diagnosis_pipe_dc_fe = load_pkl_file(
        f'outputs/ml_pipeline/predict_diagnosis/{version}/clf_pipeline_data_cleaning_feat_eng.pkl')
    diagnosis_pipe_model = load_pkl_file(
        f"outputs/ml_pipeline/predict_diagnosis/{version}/clf_pipeline_model.pkl")
    diagnosis_feat_importance = plt.imread(
        f"outputs/ml_pipeline/predict_diagnosis/{version}/features_importance.png")
    X_train = pd.read_csv(
        f"outputs/ml_pipeline/predict_diagnosis/{version}/X_train.csv")
    X_test = pd.read_csv(
        f"outputs/ml_pipeline/predict_diagnosis/{version}/X_test.csv")
    y_train = pd.read_csv(
        f"outputs/ml_pipeline/predict_diagnosis/{version}/y_train.csv").values
    y_test = pd.read_csv(
        f"outputs/ml_pipeline/predict_diagnosis/{version}/y_test.csv").values

    st.write("### ML Pipeline: Predict Breast Cancer Diagnosis")
    # display pipeline training summary conclusions
    st.info(
        f"The pipeline was tuned aiming at least 90% Recall for Malignant and 90% Precision for Benign cases. \n\n"
        f"* The model was trained on {len(X_train)} samples and tested on {len(X_test)} samples.\n\n"
        f"* The pipeline performance on train and test set is - \n"
        f"   * Benign precision is 100% for Train set and 97% for Test set\n"
        f"   * Malignant recall is 100% for Train set and 96% for Test set"
        )

    # show pipelines
    st.write("---")
    st.write("#### There are 2 ML Pipelines arranged in series.")

    st.write(" * The first is responsible for data cleaning and feature engineering.")
    st.write(diagnosis_pipe_dc_fe)

    st.write("* The second is for feature scaling and modelling.")
    st.write(diagnosis_pipe_model)

    # show feature importance plot
    st.write("---")
    st.write("* The features the model was trained and their importance.")
    st.write(X_train.columns.to_list())
    st.image(diagnosis_feat_importance)

    # evaluate performance on train and test set
    st.write("---")
    st.write("### Pipeline Performance")
    clf_performance(X_train=X_train, y_train=y_train,
                    X_test=X_test, y_test=y_test,
                    pipeline=diagnosis_pipe_model,
                    label_map=["Benign", "Malignant"])
