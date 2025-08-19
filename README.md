# Breast Cancer Diagnosis

---

![Responsive Dashboard](/static/images/dashboard.png)

This project is part of the five milestone projects within the Full Stack Developer course offered by Code Institute. It is the final project in this course and represents my chosen path in Predictive Analytics. The initial concept for this project revolves around 'working with data'.

In this project, you will be guided step by step through the entire process, from data cleaning to feature engineering. The content has been personalized to create a welcoming atmosphere, helping you gain a thorough understanding of each individual step, including what I did and how I accomplished it.

If you ever feel confused, please refer back to the README file, where you will find a wealth of important information relevant to the project.

The live application can be found [here].

## Dataset Content

The dataset is sourced from [Kaggle](https://www.kaggle.com/datasets/yasserh/breast-cancer-dataset).

**What is Kaggle?**

- Kaggle is an online community platform for data scientists and machine learning enthusiasts.
- Kaggle allows users to collaborate with other users, find and publish datasets, use GPU integrated notebooks, and compete with other data scientists to solve data science challenges.

In this project, I created a fictional user story. However, the predictive analytics conducted could be applied to a real project in the workplace.

### About the dataset

Breast cancer is the most common cancer amongst women in the world. It accounts for 30% of all cancer cases, and affected over 2.3 Million people in 2024 alone. It starts when cells in the breast begin to grow out of control. These cells usually form tumors that can be seen via X-ray or felt as lumps in the breast area.

This dataset contains 569 patient records (rows) with 32 features (columns) representing tumor characteristics extracted from digitized breast mass images. It comprises the mean, SE, and worst versions of each of the 10 core measurements. The data was collected from clinical imaging and biopsy results at the University of Wisconsin Hospitals.

The dataset includes tumor profile measurements such as:

#### Physical Features

- Tumor size (radius, perimeter, area)

- Structural properties (concavity, compactness, symmetry)

- Texture characteristics

#### Statistical feautures

- Mean values
  
- Standard errors
  
- "Worst" measurements (most severe observations)

#### Target Variable

`diagnosis`: Binary classification of tumors:

- M = Malignant (cancerous)

- B = Benign (non-cancerous)

In any part of the project where you don’t understand one of the variables used in the analysis, please refer to the table below.
Ordering starts from 0 to match the imported dataset.

### Abbreviations explained

|Variable|Meaning|Units/Categories|
|:----|:----|:----|
|0. id|Patient identification number|Integer|
|1. diagnosis|Tumor classification|M = Malignant, B = Benign|
|2. radius_mean|Mean radius of tumor nuclei|Continuous (unitless pixel or scaled)|
|3. texture_mean|Mean of standard deviation of gray-scale values|Continuous|
|4. perimeter_mean|Mean size of tumor boundary|Continuous|
|5. area_mean|Mean tumor area|Continuous|
|6. smoothness_mean|Mean local variation in radius lengths|Continuous|
|7. compactness_mean|Mean (perimeter² / area − 1.0)|Continuous|
|8. concavity_mean|Mean severity of concave portions of contour|Continuous|
|9. concave points_mean|Mean number of concave portions|Continuous|
|10. symmetry_mean|Mean symmetry of nucleus|Continuous|
|11. fractal_dimension_mean|Mean “coastline approximation” of boundary complexity|Continuous|
|12. radius_se|Standard error of radius|Continuous|
|13. texture_se|Standard error of texture|Continuous|
|14. perimeter_se|Standard error of perimeter|Continuous|
|15. area_se|Standard error of area|Continuous|
|16. smoothness_se|Standard error of smoothness|Continuous|
|17. compactness_se|Standard error of compactness|Continuous|
|18. concavity_se|Standard error of concavity|Continuous|
|19. concave points_se|Standard error of concave points|Continuous|
|20. symmetry_se|Standard error of symmetry|Continuous|
|21. fractal_dimension_se|Standard error of fractal dimension|Continuous|
|22. radius_worst|Worst (largest) radius|Continuous|
|23. texture_worst|Worst texture|Continuous|
|24. perimeter_worst|Worst perimeter|Continuous|
|25. area_worst|Worst area|Continuous|
|26. smoothness_worst|Worst smoothness|Continuous|
|27. compactness_worst|Worst compactness|Continuous|
|28. concavity_worst|Worst concavity|Continuous|
|29. concave points_worst|Worst concave points|Continuous|
|30. symmetry_worst|Worst symmetry|Continuous|
|31. fractal_dimension_worst|Worst fractal dimension|Continuous|

## Agile methodology - Development

### User Stroies

- In the beginning of the project I decided to create a Kanban project, where to input 'issues', the idea was to help me in following a
direction while building this project.
- The kanban board for this project can be found in this url [@taz1003's cancer diagnosis project](https://github.com/users/taz1003/projects/5/views/1).

![PP5 Kanban](/static/images/kanban.png)

## Crisp-DM, what is it and how is it used?

CRISP-DM, which stands for CRoss Industry Standard Process for Data Mining, is a process model that serves as the foundation for data science projects.

CRISP-DM consists of six sequential phases:

1. **Business Understanding** - What are the business requirements?
2. **Data Understanding** - What data do we have or need? Is the data clean?
   - Remember, "garbage in, garbage out," so it’s essential to ensure your data is properly cleaned.
3. **Data Preparation** - How will we organize the data for modeling?
4. **Modeling** - Which modeling techniques should we use?
5. **Evaluation** - Which model best aligns with the business objectives?
6. **Deployment** - How will stakeholders access the results?

For a more in-depth understanding of each phase and how to implement them, please refer to [CRISP-DM](https://www.datascience-pm.com/crisp-dm-2/).

## Business Case Overview

As a Data Practitioner working with Code Institute, I was approached by a leading healthcare provider organization specializing in oncology to deliver actionable insights and predictive solutions. The client aims to improve diagnostic accuracy, optimize treatment prioritization, and enhance patient care outcomes by leveraging historical diagnostic data from breast cancer screenings.

When defining the ML business case, it was agreed that the performance metric is  at least 90% Recall for Malignant and 90% Precision for Benign cases, since the client needs to detect a malignant case.
The client doesn't want to miss a malignant case, even if that comes with a cost where you misidentify a benign tumour, and state it is malignant. For this client, this is not as bad as misidentifying a malignant tumour as benign.

1. The client is interested in understanding the key diagnostic features most strongly correlated with malignant tumors so that oncologists can focus on the most relevant indicators during patient evaluations.
2. The client is interested in determining whether a newly detected tumor is malignant or benign. If malignant, the client is also interested in identifying the severity group (cluster) based on historical patient patterns. Using these insights, the client expects recommendations on the most critical diagnostic factors to monitor and strategies to improve early detection and intervention for high-risk cases.

The client also presented the criteria for the performance goal of the predictions:

- At least 90% Recall for Malignant and 90% Precision for Benign cases.
- Clustering validation with Silhouette Score ≥ 0.4 for meaningful group separation.

The client has access to a publicly available dataset containing detailed breast cancer diagnostic measurements, including tumor size, texture, shape, and other cell nucleus characteristics, along with confirmed classifications of each case as malignant or benign.

## Hypothesis and how to validate?

### Hypothesis One

***"Higher tumor size and area (area_worst) significantly increase the likelihood of malignant diagnosis."***

- A Correlation study (Pearson/Spearman + Multivariate analysis)  can help in this investigation.

### Hypothesis Two

***"Higher concavity and perimeter features are strongly linked with malignant diagnosis."***

- A correlation study (Pearson/Spearman + Multivariate analysis) and Cluster Analysis can help in investigating if this is true.

## Rationale to map the business requirements to the Data Visualizations and ML tasks

### Business Requirement 1 - **Data Visualization and Correlation Study**

As a data practitioner I will -

1. Identify the most important features (e.g., radius, area value, concavity) correlated with malignant tumors.
2. Conduct a correlation (Pearson and Spearman) and Multiviriate (MVA) study so I can better understand how the variables are correlated to cancer diagnosis, which enables me to discover how the tumor features correlate with breast cancer diagnosis.

### Outcome of Correlation study

- Higher worst area value might point to a Malignant diagnosis.

![area_worst](/static/images/cor-study-one.png)

- Mean of the concave points if >0.05 might point to a Malignant diagnosis.

![concave_points_mean](/static/images/cor-study-two.png)

- Concave worst area value if >0.14 might point to a Malignant diagnosis.

![concave_points_worst](/static/images/cor-study-three.png)

- A mean tumor boundary(perimeter) value of >85 might point to a Malignant diagnosis.

![preimeter_mean](/static/images/cor-study-four.png)

- A >100 value of outer perimeter of lobes might point to a Malignant diagnosis.

![perimeter_worst](/static/images/cor-study-five.png)

- Higher worst radius value might point to a Malignant diagnosis.

![redius_worst](/static/images/cor-study-six.png)

### Outcome of the Multiviriate Analysis (MVA)

- Multivariate analysis (MVA) is a set of statistical methods used to analyze data sets with multiple variables, examining relationships and patterns among them. We will visualize the MVA among the variables, all in one go, with a pairplot figure.

![MVA](/static/images/mvr_study.png)

### Business Requirement 2 - **Classification, Clustering, and Data Analysis**

1. To predict whether a new patient’s tumor is malignant or benign, I will build a binary classification model for this task with the most important features.
2. I want to identify the cluster profile of a new patient case to recommend potential breast cancer diagnosis and support earlier, more accurate detection.
3. I will put Clustered patients into three risk groups for required treatment procedures - High, Low & Moderate to High-Risk factors.

### Outcome of the Prediction and Cluster Analysis

#### Binary Classification

- The most important features that I found after the assessment are - "area_mean", "smoothness_worst", "perimeter_se", "texture_worst" and "fractal_dimension_se", with which I built the Binary Classification Model to predict malignant or benign tumors.

![Important Features](/static/images/imp_features.png)

#### Cluster Analysis

I achieved a silhoutte score of 49% which surpassed that client' criteria of success.
Also the cluster classifications are as follows:

- Cluster 0: Mostly malignant (69%), with moderately irregular contours (concavity 0.112-0.198), mid-range tumor size (perimeter 97.692-119.7), and high structural complexity (fractal dimension 0.104-0.124). The 31% benign cases suggest some borderline tumors in this group.

- Cluster 1: Predominantly benign (92%), featuring smooth margins (concavity 0.02-0.059), compact size (perimeter 79.7-98.385), and low structural disorder (fractal dimension 0.069-0.083). The 8% malignant cases may represent early-stage or misclassified tumors.

- Cluster 2: Entirely malignant (100%), showing highly irregular shapes (concavity 0.125-0.219), large tumor spread (perimeter 140.35-170.15), and moderate structural chaos (fractal dimension 0.076-0.092). This cluster clearly represents aggressive, high-risk malignancies.

![Clusters](/static/images/clusters.png)

#### Predictve Analysis

- After a thorough study with the classification model and cluster analysis, I was able to present an AI-based Breast Cancer Diagnosis Predictor in the Dashboard which predicts the Malignancy or Benign level of tumors in a patient's body.

![Predictor](/static/images/page_3_predictor.png)

## ML Business Case

### Business Case Assessment

- What are the business requirements?

  - Answers presented in the `Rationale to map the business requirements to the Data Visualizations and ML tasks` section.

- Is there any business requirement that can be answered with conventional data analysis?

  - Yes. Correlation analysis and visualizations can be used to investigate how tumor features (e.g., area_mean, concavity_mean, perimeter_worst) are associated with benign vs. malignant diagnoses.

- Does the client need a dashboard or an API endpoint?

  - The client needs a dashboard (already implemented with 6 pages: project summary, diagnosis study, diagnosis predictor, hypotheses, ML pipelines, clustering).

- What does the client consider as a successful project outcome?

  - A study showing the most relevant tumor attributes correlated to malignant or benign diagnosis.
  - A predictive model (with probability up to 100%) that can classify new patient cases and assign them to cluster groups.
  - Clear interpretations of the hypotheses derived from the data.
  - A deployed interactive dashboard where clinicians/researchers can visualize and experiment with inputs.

- Can you break down the project into Epics and User Stories?

  - Information gathering and data collection
  - Data visualization, cleaning, and preparation
  - Model training, optimization and validation
  - Dashboard planning, designing, and development
  - Dashboard deployment and release

- Ethical or Privacy concerns?

  - No. The dataset is publicly available (Breast Cancer Wisconsin dataset) in [Kaggle](https://www.kaggle.com/datasets/yasserh/breast-cancer-dataset).

- Does the data suggest a particular model?

  - Yes. The data suggests classification models (target = diagnosis: Benign vs. Malignant).
  - For feature relevance, tree-based models such as AdaBoostClassifier or GradientBoostingClassifier are appropriate.

- What are the model's inputs and intended outputs?

  - **Inputs**: Tumor attributes (area_mean, smoothness_worst, perimeter_se, concavity_mean, etc.).
  - **Outputs**:
    1. Diagnosis prediction (Benign or Malignant) with probability up to 100%.
    2. Cluster assignment (Cluster 0 = 69% malignant, Cluster 1 = 92% benign, Cluster 2 = 100% malignant).

- What are the criteria for the performance goal of the predictions?

  - At least 90% Recall for Malignant and 90% Precision for Benign cases which I surpassed with Benign precision at 100% for Train set, 97% for Test set and Malignant recall at 100% for Train set and 96% for Test set
  - Clustering validation with Silhouette Score ≥ 0.4 for meaningful group separation which I also surpassed with a silhoutte score of 49%.

- How will the client benefit?

  - The client (healthcare provider organization) will be able to:
    1. Understand which tumor attributes are most critical for diagnosis.
    2. Predict patient diagnosis with high probability using the interactive dashboard.
    3. Explore clusters of patients that may correspond to different biological subtypes of cancer.
    4. Test hypotheses and validate medical insights interactively.

## Dashboard Design (Streamlit App User Interface)

### Page 1: Quick project summary

Quick project summary:

- **Project Terms & Jargon**

  - Describe the breast cancer dataset shortly.

- **Project Repository link**

  - Url directing the user to this repository.

- **State Business Requirements**

  - State business requirements:
    1. Identify the most important tumor attributes correlated to diagnosis.
    2. Build ML pipelines to predict diagnosis and perform clustering analysis.

![Page 1](/static/images/page_1_summary.png)

### Page 2: Breast Cancer Diagnosis Study

- Load the data used for this project.
- Display the variables that bear the strongest correlation to diagnosis
- Checkbox: Data inspection on the breast cancer dataset.
- Describe and visualize the correlation study
among the variables and the target (diagnosis).
- Checkbox: Individual plots showing how diagnosis correlates with each key feature.

![Page 2](/static/images/page_2_study.png)

### Page 3: Diagnosis Predictor Interface

- Predict diagnosis (Benign vs. Malignant) for new patient data.
- Allow input of patient tumor features via Streamlit widgets.
- Display the predicted diagnosis probability (up to 100%).
- Assign and display cluster membership (e.g., Cluster 0 = 69% malignant, Cluster 1 = 92% benign, Cluster 2 = 100% malignant).
- Suggest appropriate treatment suggestions based on the cluster group.

![Page 1](/static/images/page_3_predictor.png)

### Page 4: Project Hypothesis and Validation

- Before the analysis, this page was designed to describe each hypothesis and its validation method.
- After the analysis I can report that:
  1. Higher tumor size and area (area_mean, area_worst) significantly increase the likelihood of malignant diagnosis.
  2. Higher concavity and perimeter features are strongly linked with malignant diagnosis.

![Page 1](/static/images/page_4_hypothesis.png)

### Page 5: ML: Predict Breast Cancer Diagnosis

- Considerations and conclusions after the pipeline training.
- Present the ML pipeline steps:
  1. The first is responsible for data cleaning and feature engineering.
  2. The second is for feature scaling and modelling.
- Enlist the features the model was trained and their importance.
- Present pipeline performace through confusion matrix using the Train and Test datasets.

![Page 1](/static/images/page_5_ml_diagnosis.png)

### Page 6: ML: Cluster Analysis

- Describe the pipeline used for cluster analysis.
- List the features used as well.
- Display results of clustering with 3 clusters with Silhoutte Score & interactive plots.
- Map clusters to diagnosis probabilities (Cluster 0 = 69% malignant, Cluster 1 = 92% benign, Cluster 2 = 100% malignant).
- Conclusion: Clusters provide additional insights into subgroups of patients to aid in treatment assessments.

![Page 1](/static/images/page_6_ml_cluster.png)

---

## Bug Fixes

### Recreating Repository Due to Corruption

- Previous repository of this project was internally corrupted which I found out while deploying to Heroku.
- Mainly because, the versions of the requirements.txt packages were not compatible along with incompatibilty issues with
Python 3.12.1, which crashed and corrupted the whole repository.
- I recreated the repository of the project and transferrred the files and made sure to keep the package files compatible.
- One down side of this is the lesser number of commits and bug fixes I originally had.
- The previous repo is in this [GitHub link](https://github.com/taz1003/breast-cancer-diagnosis-PP5), to check out if interrested.

### Data Cleaning Notebook (Previous Redo)

- This bug fix was from the previous [repository](https://github.com/taz1003/breast-cancer-diagnosis-PP5).
- During the correlation and PPS study, after I ran the `CalculateCorrAndPPS(df)` function, I got a warning that denotes - "`FutureWarning: is_categorical_dtype is deprecated and will be removed in a future version. Use isinstance(dtype, CategoricalDtype) instead`".
- After a brief online research and discussions with my peers, I got rid of the warning by adding `_is_categorical_dtype(series)` function before running the `CalculateCorrAndPPS(df)` function.

### Cluster Notebook (1)

- This bug fix was from the previous [repository](https://github.com/taz1003/breast-cancer-diagnosis-PP5).
- During the process of finding the optimized values of the clusters using Elbow Method and Silhoutte Score, I got font-waarning - `findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial, Liberation Sans, Bitstream Vera Sans, sans-serif findfont: Font family ['sans-serif'] not found. Falling back to DejaVu Sans.`
- Fixed the font-issue warning by specifying the fonts taken from [StackOverflow](https://stackoverflow.com/questions/42097053/matplotlib-cannot-find-basic-fonts).

### Cluster Notebook (2)

- This bug fix was from the previous [repository](https://github.com/taz1003/breast-cancer-diagnosis-PP5).
- During the assessment of the most important features, that define a cluster, I was getting an error - `The 'Pipeline' has no attribute 'transform'`.
- The issue was because the pipeline `PipelineClf2ExplainClusters` ends with a classifier `GradientBoostingClassifier` and Scikit-learn’s Pipeline.transform() only works if all final steps have transform() methods.
- With the help of [StackOverflow](https://stackoverflow.com/questions/57043168/attribute-error-pipeline-object-has-not-attribute-transform) and [Scikit-learn](https://scikit-learn.org/stable/modules/feature_selection.html), fixed the issue by adding the features after scaling and feature selection, but before the classifier and fitting them into a variable.

### Cluster Notebook (3)

- This bug fix was from the previous [repository](https://github.com/taz1003/breast-cancer-diagnosis-PP5).
- During the cluster analysis based on their profiles, I ran into an error that said `AttributeError: 'DataFrame' object has no attribute 'append'`.
- This happended because `DataFrame.append()`, which was used in the `DescriptionAllClusters()` function, was deprecated in the previous Pandas versions than the one i am using.
- Fixed it by using the modern replacement `pd.concat` to concatenate `DescriptionAllClusters`, `ClusterDescription`.

### Page 6 Cluster (Dashboard)

- While deploying to Heroku, I ran into this error `ValueError: Mime type rendering requires ipython but it is not installed` in
Page 6 (Cluster) of the Dashboard.
- Fixed it by replacing `fig.show(renderer='jupyterlab')` with `st.plotly_chart(fig)`.

## Unfixed Bugs

- In Page 5 - ML: Predict Breast Cancer Diagnosis from the Dashboard, the Train and Test tables contains
Previous Benign and Previous Malignant texts a bit disrupted with brackets and commas.

## Deployment

The main branch of this repository has been used for the deployed version of this application.

### Using Github & VSCode

To deploy my Data application, I used the [Code Institute milestone-project-bring-your-own-data Template](https://github.com/Code-Institute-Solutions/milestone-project-bring-your-own-data).

- Click the 'Use This Template' button.
- Add a repository name and brief description.
- Click the 'Create Repository from Template' to create your repository.
- To create a workspace you then need to click 'Code', then 'Create codespace on main', this can take a few minutes.
- When you want to work on the project it is best to open the workspace from 'Codespaces' as this will open your previous workspace rather than creating a new one. You should pin the workspace so that it isn't deleted.
- Committing your work should be done often and should have clear/explanatory messages, use the following commands to make your commits:
  - `git add .`: adds all modified files to a staging area
  - `git commit -m "A message explaining your commit"`: commits all changes to a local repository.
  - `git push`: pushes all your committed changes to your Github repository.

### Forking the GitHub Repository

By forking the GitHub Repository you will be able to make a copy of the original repository on your own GitHub account allowing you to view and/or make changes without affecting the original repository by using the following steps:

1. Log in to GitHub and locate the [GitHub Repository](repo here???)
2. At the top of the Repository (not top of page) just above the "Settings" button on the menu, locate the "Fork" button.
3. You should now have a copy of the original repository in your GitHub account.

### Making a Local Clone

1. Log in to GitHub and locate the [GitHub Repository](https://github.com/taz1003/breast-cancer-diagnosis-PP5)
2. Under the repository name, click "Clone or download".
3. To clone the repository using HTTPS, under "Clone with HTTPS", copy the link.
4. Open commandline interface on your computer
5. Change the current working directory to the location where you want the cloned directory to be made.
6. Type `git clone`, and then paste the URL you copied in Step 3. `$ git clone (paste url)`
7. Press Enter. Your local clone will be created.

### Deployment To Heroku

- The App live link is: (paste url)
- The project was deployed to Heroku using the following steps.

1. Log in to Heroku and create an App
2. At the Deploy tab, select GitHub as the deployment method.
3. Select your repository name and click Search. Once it is found, click Connect.
4. Select the branch you want to deploy, then click Deploy Branch.
5. The deployment process should happen smoothly in case all deployment files are fully functional. Click now the button Open App on the top of the page to access your App.

---

## Main Data Analysis and Machine Learning Libraries

|Libraries Used In The Project|How I Used The Library|Link|
|:----|:----|:----|
|Numpy|Used to process arrays that store values and aka data|[URL](https://numpy.org/)|
|Pandas|Used for data analysis, data exploration, data manipulation,and data visualization|[URL](https://pandas.pydata.org/)|
|Matplotlib|Used for graphs and plots to visualize the data|[URL](https://matplotlib.org/)|
|Seaborn|Used to visualize the data in the Streamlit app with graphs and plots|[URL](https://seaborn.pydata.org/)|
|ML: feature-engine|Used for engineering the data for the pipeline|[URL](https://feature-engine.readthedocs.io/en/latest/)|
|ML: Scikit-learn|Used to creat the pipeline and apply algorithms, and feature engineering steps|[URL](https://scikit-learn.org/stable/)|
|Streamlit|Used for creating the app to visualize the project's study|[URL](https://streamlit.io/)|
|Kaggle|Used to import the dataset required to perform the analysis|[URL](https://www.kaggle.com/)|
|Grammarly|Used to improve, modify or add written communications throughout the project|[URL](https://app.grammarly.com/)|

## Credits & Content

- The content of this project, represent the understanding provided by walk-through projects provided by Code Institute.
There might be some similarities as some contents have been taken and modified directly from the walk-through project 2 'Churnometer'.

- [Kaggle](https://www.kaggle.com/datasets/yasserh/breast-cancer-dataset) provided the dataset used in this project.

- Some bugs and issues appeared while the project was under contruction which have been fixed using [Stack Overflow](https://stackoverflow.com/questions).

- Got the idea for the best model_n_estimators for AdaBoostClassifier in the Predict Diagnosis notebook from [Stack Overflow](https://stackoverflow.com/questions/47216224/selecting-n-estimators-based-on-dataset-size-for-adaboostclassifier)

- I have explored in more details different terms used within deep machine learning from the Youtube channel - [Infinite Codes](https://www.youtube.com/@InfiniteCodes_/featured).

- The readme file was built using the Code Institute template.

- Some elements (README.md and Jupyter Notebook) presented throughout the project have been inspired from [Van-essa](https://github.com/van-essa/heritage-housing-issues).

- My Mentor Gareth_Mentor who guided me through the project making sure the best practices were used.

- My peer [Moshiur Rahman](https://www.linkedin.com/in/mrahman2352k/) for providing some ideas, including picking out the dataset, throughout this project.

### Media

- The [Am I Responsive](https://ui.dev/amiresponsive) page was used to get the introductory image in README.md.
- The icon for the deployed project was taken from [Twemoji](https://twemoji-cheatsheet.vercel.app/).

## Acknowledgements

- A big thank you to Code Institute for such an amazing course! I've really enjoyed every moment of this experience, and it's been awesome to see my mindset evolve as I learn and interact with the material.

- Many thanks to my mentor Gareth_Mentor who guided me through the project making sure the best practices were used.

- Appreciation towards [Moshiur Rahman](https://www.linkedin.com/in/mrahman2352k/) for providing support for this project.

- Huge thanks to my peers at Slack for providing me ideas and courage to go through this course.
