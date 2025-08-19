import streamlit as st
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix


def confusion_matrix_and_report(X, y, pipeline, label_map):

    prediction = pipeline.predict(X)

    st.write(pd.DataFrame(
        confusion_matrix(y_true=y, y_pred=prediction),
        columns=[["Predicted " + sub for sub in label_map]],
        index=[["Actual " + sub for sub in label_map]]
    ))
    st.write("\n")

    st.write('---  Classification Report  ---')
    st.write(classification_report(y, prediction, target_names=label_map), "\n")


def clf_performance(X_train, y_train, X_test, y_test, pipeline, label_map):
    st.write("#### Train Set #### \n")
    confusion_matrix_and_report(X_train, y_train, pipeline, label_map)

    st.write("#### Test Set ####\n")
    confusion_matrix_and_report(X_test, y_test, pipeline, label_map)
