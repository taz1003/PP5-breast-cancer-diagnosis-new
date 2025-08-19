# The code below was inspired by the Churnometer Project from Code Institute 
# with some adjustments

import streamlit as st


def page_summary_body():

    st.write("### Quick Project Summary")

    # text based on README file - "Dataset Content" section
    """Project Dataset"""
    st.info(
        f"**Project Dataset & Jargons**\n"
        f" * The dataset is sourced from **[Kaggle](https://www.kaggle.com/datasets/yasserh/breast-cancer-dataset)**. "
        f" This dataset contains 569 patient records (rows) with 32"
        f" features (columns) representing tumor characteristics extracted"
        f" from digitized breast mass images.\n"
        f"* The dataset includes mainly numerical features of tumor"
        f" characteristics, such as **radius**, **texture**, **smoothness**,"
        f" **concavity**, **compactness**, **concave points**, & **symmetry**."
        f" The target variable is **diagnosis**, indicating whether the tumor"
        f" is malignant (M) or benign (B).\n"
        f"* The dataset is well-suited for classification tasks, particularly"
        f" in the context of breast cancer diagnosis."
        f" There is also clustering information available, which can be useful"
        f" for exploration of the severity of tumors."
    )

    # Link to README file, so the users can have access to full project 
    # documentation
    st.write("### Project Repository")
    st.write(
        f"The code and resources for this project are available on GitHub, "
        f"for additional information, please visit and **read** the "
        f"[Project README file](https://github.com/taz1003/PP5-breast-cancer-diagnosis-new)."
    )

    # "Business Requirements" section
    st.write("###  Business Requirements")
    st.success(
        f"The project has 2 business requirements.\n"
        f"The client, a leading healthcare provider organization specializing in oncology,"
        f" has presented the following requirements:\n"
        f"1. Requirement 1 - The client is interested in understanding the key "
        f"diagnostic features most strongly correlated with malignant tumors "
        f"so that oncologists can focus on the most relevant indicators during patient evaluations.\n"
        f"2. Requirement 2 - The client is interested in determining whether a newly detected tumor "
        f"is malignant or benign. If malignant, "
        f"the client is also interested in identifying the severity group (cluster) based on historical patient patterns. \n"
        f"Using these insights, the client expects recommendations on the most critical diagnostic factors to monitor "
        f"and strategies to improve early detection and intervention for high-risk cases."
    )
