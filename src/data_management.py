# code copied from Code Institute's Churnornmeter Project with some adjustments
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Function to load the breast cancer dataset


@st.cache_data
def load_breast_cancer_data():
    df = pd.read_csv("outputs/datasets/collection/breast-cancer.csv")
    return df


def load_pkl_file(file_path):
    return joblib.load(filename=file_path)
