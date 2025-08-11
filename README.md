# Burnout Risk Prediction System

## Problem Background
Workplace burnout is a psychological syndrome caused by prolonged, unmanaged stress. It is characterized by emotional exhaustion, detachment, and reduced professional performance. While not classified as a medical condition, the World Health Organization (WHO) recognizes burnout as an occupational phenomenon in the ICD-11.

Burnout impacts both individual well-being and organizational performance. It increases the risk of depression, anxiety, absenteeism, disengagement, and turnover. A 2023 Gallup report found that 76% of employees experience burnout at least occasionally, with 21% reporting they feel burned out “very often or always.”

For companies, burnout leads to reduced productivity, higher healthcare costs, and expensive employee turnover. It also undermines morale, team cohesion, and long-term success. Despite these consequences, many organizations still lack systems for early detection and intervention, missing critical opportunities for prevention and support.

Research published in the American Journal of Preventive Medicine quantifies these losses:

* $3,999 per nonmanagerial hourly employee annually

* $4,257 per nonmanagerial salaried employee annually

* $10,824 per manager annually

* $20,683 per executive annually

**For an average U.S. company with 1,000 employees, this totals $5.04 million in annual losses**, alongside a significant reduction in overall employee quality of life. These statistics highlight the urgent need for early detection and prevention strategies.

## Project Output
This project develops a machine learning system to support mental health monitoring in the workplace, with three main objectives:

1. Predict burnout risk based on factors including:

* Psychological well-being

* Workplace environment and workload

* Work-life balance and lifestyle habits

2. Provide personalized recommendations based on prediction outcomes:

* At risk: Suggest immediate professional support (e.g., therapy with a licensed psychologist) and provide resources for managing work-related stress.

* Not at risk: Offer preventive materials on early warning signs, stress-reduction strategies, and optional professional support access.

3. Promote workplace mental health awareness by enabling proactive monitoring and early intervention strategies.

## Data
The dataset is a synthetic online survey simulating the impact of remote work on mental health and burnout.

* Observations: 4,500

* Missing Values: None

* Features: Numerical, ordinal, and categorical variables

* Target Variable: Binary (burnout risk: 1 = at risk, 0 = not at risk)

* Class Imbalance Handling: SMOTENC applied for balanced training data.

## Method
The task is a binary classification problem, evaluated using macro F1 score to ensure balanced performance across both classes.

**Best Model**: GradientBoostingClassifier

**Approach:**

* Data preprocessing with scaling, encoding, and oversampling (SMOTENC)

* Model training and evaluation with cross-validation

* Hyperparameter tuning using GridSearchCV

## Stacks
**Data Processing & Feature Engineering**

* pandas

* numpy

* feature_engine

* scipy

* scikit-learn (ColumnTransformer, StandardScaler, OneHotEncoder, OrdinalEncoder)

* imbalanced-learn (SMOTENC)

**Modeling**

* scikit-learn (KNeighborsClassifier, SVC, DecisionTreeClassifier, RandomForestClassifier, GradientBoostingClassifier)

**Evaluation & Metrics**

* scikit-learn (classification_report, f1_score, ConfusionMatrixDisplay, cross_val_score, GridSearchCV)

**Visualization**

* seaborn

* matplotlib

**Deployment**

* streamlit

