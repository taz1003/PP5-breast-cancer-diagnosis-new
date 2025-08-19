# Breast Cancer Diagnosis

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

- In the beginning of the project I decided to create a Kanban project, where to input 'issues', the idea was to help me in following a
direction while building this project.
- The kanban board for this project can be found in this url [@taz1003's cancer diagnosis project](https://github.com/users/taz1003/projects/5).

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

The client has access to a publicly available dataset containing detailed breast cancer diagnostic measurements, including tumor size, texture, shape, and other cell nucleus characteristics, along with confirmed classifications of each case as malignant or benign.

## Rationale to map the business requirements to the Data Visualizations and ML tasks

### Business Requirement 1 - **Data Visualization and Correlation Study**

As a data practitioner I will -

1. identify the most critical features (e.g., radius, area value, concavity) correlated with malignant tumors.
2. use statistical and visual analysis to guide early diagnosis.

### Business Requirement 2 - **Classification, Clustering, and Data Analysis**

1. Predict whether a new patient’s tumor is malignant or benign. I will build a binary classification model for this task.
2. I want to identify the cluster profile of a new patient case to recommend potential diagnostic focus areas and support earlier, more accurate detection.
3. Cluster patients into risk groups for personalized treatment plans.

## Hypothesis and how to validate?

### Hypothesis One

***"Tumors with a larger worst area and greater mean perimeter (>85) are more likely to be Malignant."***

- A Correlation study (Pearson/Spearman + Multivariate analysis)  can help in this investigation.

### Hypothesis Two

***"Tumors with higher concavity_mean and concavity_worst values are more likely to be Malignant."***

- A correlation study (Pearson/Spearman + Multivariate analysis) can help in investigating if this is true.

## ML Business Case

### Binary Classification (Malignant vs. Benign)

### Severity Estimation (Regression?)

### Patient Clustering (Unsupervised Learning)

## Dashboard Design (Streamlit App User Interface)

### Page 1: Quick project summary

Quick project summary:

- Project Terms & Jargon
- Describe Project Dataset
- State Business Requirements

---

## Bugs

### Data Cleaning Notebook

- During the correlation and PPS study, after I ran the `CalculateCorrAndPPS(df)` function, I got a warning that denotes - "`FutureWarning: is_categorical_dtype is deprecated and will be removed in a future version. Use isinstance(dtype, CategoricalDtype) instead`".
- After a brief online research and discussions with my peers, I got rid of the warning by adding `_is_categorical_dtype(series)` function before running the `CalculateCorrAndPPS(df)` function.

### Cluster Notebook (1)

- During the process of finding the optimized values of the clusters using Elbow Method and Silhoutte Score, I got font-waarning - `findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial, Liberation Sans, Bitstream Vera Sans, sans-serif findfont: Font family ['sans-serif'] not found. Falling back to DejaVu Sans.`
- Fixed the font-issue warning by specifying the fonts taken from [StackOverflow](https://stackoverflow.com/questions/42097053/matplotlib-cannot-find-basic-fonts).

### Cluster Notebook (2)

- During the assessment of the most important features, that define a cluster, I was getting an error - `The 'Pipeline' has no attribute 'transform'`.
- The issue was because the pipeline `PipelineClf2ExplainClusters` ends with a classifier `GradientBoostingClassifier` and Scikit-learn’s Pipeline.transform() only works if all final steps have transform() methods.
- With the help of [StackOverflow](https://stackoverflow.com/questions/57043168/attribute-error-pipeline-object-has-not-attribute-transform) and [Scikit-learn](https://scikit-learn.org/stable/modules/feature_selection.html), fixed the issue by adding the features after scaling and feature selection, but before the classifier and fitting them into a variable.

### Cluster Notebook (3)

- During the cluster analysis based on their profiles, I ran into an error that said `AttributeError: 'DataFrame' object has no attribute 'append'`.
- This happended because `DataFrame.append()`, which was used in the `DescriptionAllClusters()` function, was deprecated in the previous Pandas versions than the one i am using.
- Fixed it by using the modern replacement `pd.concat` to concatenate `DescriptionAllClusters`, `ClusterDescription`.

## Deployment

The master branch of this repository has been used for the deployed version of this application.

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

## Credits

- Got the idea for the best model_n_estimators for AdaBoostClassifier in the Predict Diagnosis notebook from [StackOverflow](https://stackoverflow.com/questions/47216224/selecting-n-estimators-based-on-dataset-size-for-adaboostclassifier)

### Content

- The text for the Home page was taken from Wikipedia Article A
- Instructions on how to implement form validation on the Sign-Up page were taken from [Specific YouTube Tutorial](https://www.youtube.com/)
- The icons in the footer were taken from [Font Awesome](https://fontawesome.com/)

### Media

- The photos used on the home and sign-up page are from This Open-Source site
- The images used for the gallery page were taken from this other open-source site

## Acknowledgements (optional)

Thank the people who provided support through this project.
