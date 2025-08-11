# Milestone 2

## Repository Outline

```
 |
    ├── deployment/ -> contains all files used for model deployment
    │   ├── app.py -> serves as the homepage for EDA and prediction
    │   └── eda.py -> contains plots created in the notebook, to be displayed on the dashboard
    │   └── prediction.py -> contains prediction features from the notebook, to be displayed on the dashboard
    │   └── model.pkl -> the final trained model with tuned parameters for prediction
    ├── description.md -> documentation describing the project repository
    ├── P1M2_Aulia-Putri_conceptual.txt -> answers to conceptual questions about feature engineering in machine learning
    ├── P1M2_Aulia-Putri.ipynb -> notebook containing the full model-building process from start to finish
    ├── P1M2_Aulia-Putri_inf.ipynb -> notebook for testing and inference using the final model
    ├── url.txt -> contains the deployment dashboard URL
    └── README.md -> contains instructions and performance metrics for Milestone 2

```

## Problem Background
Workplace burnout is a psychological syndrome caused by prolonged, unmanaged stress. It is marked by emotional exhaustion, detachment, and reduced professional performance. While not classified as a medical condition, the [World Health Organization (WHO)](http://who.int/standards/classifications/frequently-asked-questions/burn-out-an-occupational-phenomenon) recognizes burnout as an occupational phenomenon in the ICD-11.

Burnout affects more than individual health, it has serious organizational consequences. It increases the risk of depression, anxiety, absenteeism, disengagement, and turnover. A [2023 Gallup](https://www.gallup.com/workplace/288539/employee-burnout-biggest-myth.aspx) report found that 76% of employees experience burnout at least occasionally, with 21% reporting they feel burned out “very often or always.”

For companies, this translates to significant losses including reduced productivity, higher healthcare costs, and expensive employee replacement. Burnout also damages morale, team dynamics, and long-term performance. Despite this, many organizations still lack systems to detect and address mental health risks early, missing critical opportunities for prevention and support. Early detection is essential—not only to protect employee well-being, but also to enable timely interventions that prevent deeper organizational impact.

Research published in the [American Journal of Preventive Medicine](https://www.ajpmonline.org/article/S0749-3797(25)00023-6/abstract) shows just how costly burnout can be. Over the course of a year, employee disengagement, overextension, and burnout cost employers an average of $3,999 per nonmanagerial hourly employee, $4,257 per nonmanagerial salaried employee, $10,824 per manager, and $20,683 per executive. For an average U.S. company with 1,000 employees, this adds up to $5.04 million in losses annually, along with a significant reduction in overall employee quality of life. These numbers highlight the urgent need for early detection and prevention strategies in the workplace.

## Project Output
This project aims to develop a machine learning system designed to support mental health monitoring in the workplace. The system has three main goals:

1. Predict an individual's risk of experiencing burnout, based on features related to:

    * Psychological well-being

    * Workplace environment and workload

    * Work-life balance and lifestyle habits

2. Provide personalized recommendations based on the prediction outcome:
    * **For individuals at risk**: The system will recommend seeking immediate professional support, such as therapy with a licensed psychologist or counselor. In addition, it will offer helpful resources and articles on how to manage work-related stress and maintain mental well-being.

    * **For individuals not at risk**: The system will provide preventive educational materials, including information on early warning signs of burnout, stress-reduction strategies, and optional access to professional support.
    
This system is designed to help companies proactively monitor and support employee well-being, while promoting a culture of mental health awareness and early intervention in the workplace.

## Data
The dataset is a synthetic online survey simulating the impact of remote work on mental health and burnout. It contains 4,500 observations with no missing values and includes a mix of numerical, ordinal, and categorical features. The target is a binary variable indicating burnout risk. The original class imbalance was addressed using SMOTENC, allowing for more balanced and robust model training.


## Method
The dataset used is  a multiclass classification task, so the most suitable evaluation metric is the macro F1 score, and the best-performing model is the GradientBoostingClassifier.


## Stacks
``` py
seaborn as sns

from feature_engine.outliers import Winsorizer

from scipy.stats import kendalltau, chi2_contingency
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline as sklPipeline
from imblearn.pipeline import Pipeline as imbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import classification_report, ConfusionMatrixDisplay,f1_score

from imblearn.over_sampling import SMOTENC

import streamlit as stimport numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import 

```


## Reference
**Referensi tambahan:**
1. http://who.int/standards/classifications/frequently-asked-questions/burn-out-an-occupational-phenomenon
2. https://www.gallup.com/workplace/288539/employee-burnout-biggest-myth.aspx
3. https://www.ajpmonline.org/article/S0749-3797(25)00023-6/abstract
4. https://pubmed.ncbi.nlm.nih.gov/36706670/
5. https://www.linkedin.com/pulse/impact-burnout-employee-productivity-call-action-modern-workplaces-5hyfc/
6. https://www.ualberta.ca/en/folio/2024/01/seven-things-you-should-know-about-job-burnout.
7. html#:~:text=He%20cites%20research%20conducted%20by,of%20harmful%20effects%20on%20health.