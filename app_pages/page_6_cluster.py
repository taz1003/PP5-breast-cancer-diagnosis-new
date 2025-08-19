# The code below was inspired by the Churnometer Project from Code Institute 
# with some adjustments
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from src.data_management import load_breast_cancer_data, load_pkl_file


def page_cluster_body():

    # load cluster analysis files and pipeline
    version = 'v1'

    cluster_pipe = load_pkl_file(
        f"outputs/ml_pipeline/cluster_analysis/{version}/cluster_pipeline.pkl")
    cluster_silhouette = plt.imread(
        f"outputs/ml_pipeline/cluster_analysis/{version}/clusters_silhouette.png")
    features_to_cluster = plt.imread(
        f"outputs/ml_pipeline/cluster_analysis/{version}/features_define_cluster.png")
    cluster_profile = pd.read_csv(
        f"outputs/ml_pipeline/cluster_analysis/{version}/clusters_profile.csv")
    cluster_features = (pd.read_csv(f"outputs/ml_pipeline/cluster_analysis/{version}/TrainSet.csv")
                        .columns
                        .to_list()
                        )

    # dataframe for cluster_distribution_per_variable()
    df_diagnosis_vs_clusters = load_breast_cancer_data().filter(['diagnosis'], axis=1)
    df_diagnosis_vs_clusters['Clusters'] = cluster_pipe['model'].labels_

    st.write("### ML Pipeline: Cluster Analysis")

    # display pipeline training summary conclusions
    st.info(
        f"* We refitted the cluster pipeline using fewer variables, and it delivered equivalent "
        f"performance to the pipeline fitted using all variables.\n"
        f"* The pipeline average silhouette score is 0.49"
    )
    st.write("---")

    st.write("#### Cluster ML Pipeline steps")
    st.write(cluster_pipe)

    st.write("#### The features the model was trained with")
    st.write(cluster_features)

    st.write("#### Clusters Silhouette Plot")
    st.image(cluster_silhouette)

    cluster_distribution_per_variable(
        df=df_diagnosis_vs_clusters, target='diagnosis')

    st.write("#### Most important features to define a cluster")
    st.image(features_to_cluster)

    # text based on "06 - Modeling and Evaluation - Cluster Sklearn" 
    # notebook conclusions
    st.write("#### Cluster Profile")

    # hack to not display the index in st.table() or st.write()
    cluster_profile.index = pd.Index([" "] * len(cluster_profile))
    st.table(cluster_profile)

    statement = (
        f"Since we have set '0' as Benign (B) and '1' as Malignant (M), from "
        f"the above profiling we can describe each clusters in the following -\n\n"
        f"* In Cluster 0, mean concavity values range from 0.112 to 0.198, indicating "
        f"moderately irregular tumor contours. The worst perimeter "
        f"measurements range from 97.692 to 119.7, suggesting mid-range tumor size, "
        f"while the worst fractal dimension falls between 0.104 and 0.124, "
        f"indicating high structural complexity. Mean perimeter values of 81.505."
        f"to 98.165 reflect moderately sized tumors. Notably, 69% of diagnoses in"
        f"this cluster are malignant (M), with 31% benign (B), suggesting a"
        f"significant risk of malignancy with some borderline cases.\n\n"
        f"* Cluster 1 shows mean concavity from 0.02 to 0.059, indicating smooth tumor margins. "
        f"The worst perimeter is significantly smaller, ranging from "
        f"79.7 to 98.385, while the worst fractal dimension (0.069 to 0.083) suggests "
        f"low structural disorder. The mean perimeter ranges from 71.85 to 87. "
        f"86, reflecting compact tumor shapes. Diagnoses in this cluster are predominantly "
        f"benign (B: 92%), with only 8% malignant (M), consistent with a low-risk profile.\n\n"
        f"* In Cluster 2, mean concavity ranges from 0.125 to 0.219, indicating highly "
        f"irregular tumor shapes. The worst perimeter spans from 140.35 to "
        f"170.15, suggesting large tumor size, while the mean perimeter varies between "
        f"115.8 and 134.7. The worst fractal dimension (0.076 to 0.092) shows "
        f"moderate structural complexity. Notably, 100% of the diagnoses in this "
        f"cluster are malignant (M), indicating an aggressive, high-risk tumor "
        f"profile that requires immediate attention. "
    )
    st.info(statement)

    statement_short = (
        f"A concise summary of the clusters:\n\n"
        f"* Cluster 0: Mostly malignant (69%), with moderately irregular "
        f"contours (concavity 0.112-0.198), mid-range tumor size (perimeter 97.692-119.7), and"
        f"high structural complexity (fractal dimension 0.104-0.124). The"
        f"31% benign cases suggest some borderline tumors in this group.\n"
        f"* Cluster 1: Predominantly benign (92%), featuring smooth margins "
        f"(concavity 0.02-0.059), compact size (perimeter 79.7-98.385), and low"
        f"structural disorder (fractal dimension 0.069-0.083). The 8%"
        f"malignant cases may represent early-stage or misclassified tumors.\n"
        f"* Cluster 2: Entirely malignant (100%), showing highly irregular shapes "
        f"(concavity 0.125-0.219), large tumor spread (perimeter 140.35-170.15,"
        f"and moderate structural chaos (fractal dimension 0.076-0.092)."
        f"This cluster clearly represents aggressive, high-risk malignancies."
    )
    st.success(statement_short)

# code coped from "06 - Modeling and Evaluation - Cluster Sklearn" notebook - 
# under "Cluster Analysis" section


def cluster_distribution_per_variable(df, target):

    df_bar_plot = df.groupby(
        ['Clusters', target]).size().reset_index(name='Count')
    df_bar_plot.columns = ['Clusters', target, 'Count']
    df_bar_plot[target] = df_bar_plot[target].astype('object')

    print(f"Clusters distribution across {target} levels")
    fig = px.bar(df_bar_plot, x='Clusters', y='Count',
                 color=target, width=800, height=500)
    fig.update_layout(xaxis=dict(tickmode='array',
                      tickvals=df['Clusters'].unique()))
    fig.show(renderer='jupyterlab')

    df_relative = (df
                   .groupby(["Clusters", target])
                   .size()
                   .unstack(fill_value=0)
                   .apply(lambda x: 100 * x / x.sum(), axis=1)
                   .stack()
                   .reset_index(name='Relative Percentage (%)')
                   .sort_values(by=['Clusters', target])
                   )

    print(f"Relative Percentage (%) of {target} in each cluster")
    fig = px.line(df_relative, x='Clusters', y='Relative Percentage (%)',
                  color=target, width=800, height=500)
    fig.update_layout(xaxis=dict(tickmode='array',
                      tickvals=df['Clusters'].unique()))
    fig.update_traces(mode='markers+lines')
    st.plotly_chart(fig)
