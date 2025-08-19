# The code below was inspired by the Churnometer Project from Code Institute
# with some adjustments
import streamlit as st


def predict_diagnosis(X_live, diagnosis_features, diagnosis_pipeline_dc_fe, diagnosis_pipeline_model):
    # from live data, subset features related to this pipeline
    X_live_diagnosis = X_live.filter(diagnosis_features)

    # apply data cleaning / feat engine pipeline to live data
    X_live_diagnosis_dc_fe = diagnosis_pipeline_dc_fe.transform(X_live_diagnosis)

    # predict
    diagnosis_prediction = diagnosis_pipeline_model.predict(X_live_diagnosis_dc_fe)
    diagnosis_prediction_proba = diagnosis_pipeline_model.predict_proba(
        X_live_diagnosis_dc_fe)
    # st.write(churn_prediction_proba)

    # Create a logic to display the results
    diagnosis_prob = diagnosis_prediction_proba[0, diagnosis_prediction][0]*100
    if diagnosis_prediction == 1:
        diagnosis_result = 'Malignant'
    else:
        diagnosis_result = 'Benign'

    statement = (
        f'### There is {diagnosis_prob.round(1)}% probability '
        f'that this patient has **{diagnosis_result}** tumor.')

    st.write(statement)

    return diagnosis_prediction


def predict_cluster(X_live, cluster_features, cluster_pipeline, cluster_profile):

    # from live data, subset features related to this pipeline
    X_live_cluster = X_live.filter(cluster_features)

    # predict
    cluster_prediction = cluster_pipeline.predict(X_live_cluster)

    statement = (
        f"### The patient is expected to belong to **cluster {cluster_prediction[0]}**")
    st.write("---")
    st.write(statement)

    # text based on "06 - Modeling and Evaluation - Cluster Sklearn" notebook conclusions
    statement = (
        f"* Analytically **patients in Clusters 0  tend to have Benign tumors** "
        f"whereas in **Cluster 1 nearly all patients have higher chance of having Malignant tumors** "
        f"and in **Cluster 2 most of the patients might have Malignant tumors**."
    )
    st.info(statement)

    # text based on "06 - Modeling and Evaluation - Cluster Sklearn" notebook conclusions
    statement = (
        f"* **Cluster 0:** Mixed but malignant-leaning (69%), with moderate sizes, irregular shapes, and the highest chaos."
        f" Patients belonging to cluster 0 are at the *moderate to high-risk* factor. Close monitoring is suggested.\n\n"
        f"* The cluster profile interpretation allowed us to label the cluster in the following fashion:\n"
        f"* **Cluster 1:** Mostly benign (92%), with smooth contours, smaller tumor sizes, and low structural disorder. "
        f"Patients belonging to cluster 1 are at the *low-risk* factor.  Regular check-ups are suggested.\n\n"
        f"* **Cluster 2:** Entirely malignant (100%), marked by large tumor size, irregular margins, and aggressive growth."
        f" Patients belonging to cluster 2 are at the *high-risk* factor. Immediate treatment is suggested.\n\n"
    )
    st.success(statement)
    
    # hack to not display index in st.table() or st.write()
    cluster_profile.index = [" "] * len(cluster_profile)
    # display cluster profile in a table - it is better than in st.write()
    st.table(cluster_profile)