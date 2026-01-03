# NYC Real Estate Price Predictor: Project Report

**Student Name:** [Louis-Marie Belfort Nzitongo Libero]
**Course:** Information Technology
**Date:** Saturday, January 3rd, 2026

---

## 🔗 Project Versions & Repository Access
**GitHub Repository:** [https://github.com/Beliro/PUIT22217146]

This project evolved through three distinct distinct stages. While this written report focuses on the **Version 2 (Random Forest)** implementation (the submitted version), the repository contains the source code for all three iterations:

| Version | Branch/Model | Accuracy (R²) | Description |
| :--- | :--- | :--- | :--- |
| **v1.0** | `master` | **0.14 (14%)** | Initial baseline attempt with raw, uncleaned data. |
| **v2.0** | `random-forest` | **0.54 (54%)** | **[SUBMITTED VERSION]** The primary model documented below. Features robust cleaning and feature engineering. |
| **v3.0** | `xgboost` | **0.61 (61%)** | Experimental advanced version using Gradient Boosting for higher precision. |

---

## 1. Introduction
The objective of this project was to develop a machine learning model capable of predicting real estate listing prices in New York City. By analyzing a dataset of listing attributes—including location, room type, and reviews—the goal was to create an automated "Real Estate Advisor" that estimates fair market value.

The project followed a standard data science pipeline: **Exploratory Data Analysis (EDA)**, **Data Cleaning**, **Feature Engineering**, and **Model Training**. This report documents the **Random Forest** implementation (Version 2).

---

## 2. Methodology

### Step 1: Exploratory Data Analysis (Visualization)
**Objective:** To understand the spatial distribution of prices and identify market patterns.

We utilized `seaborn` to plot listings on a geographic scatter plot. This confirmed that location is the primary driver of price, with distinct high-cost clusters in Manhattan.

**Code Snippet:**
```python
# Visualising the market density
sns.scatterplot(
    data=df[df['price'] < 500], 
    x='longitude', 
    y='latitude', 
    hue='neighbourhood_group'
)
```

### Step 2: Data Cleaning and Preprocessing
**Objective:** To improve data quality by removing errors and outliers that could distort predictions.

Real-world data often contains noise. We addressed two critical issues:
1.  **Imputation:** Missing values in `reviews_per_month` were filled with `0`, assuming that a lack of data implies no reviews were given.
2.  **Outlier Removal:** The dataset contained extreme outliers (e.g., luxury mansions >$5,000). We filtered the dataset to include only the general market ($10 - $500) to stabilize the model.

**Code Snippet:**
```python
# Filtering for the "General Market" range
df = df[(df['price'] > 10) & (df['price'] < 500)]
```

### Step 3: Feature Engineering
**Objective:** To transform raw data into informative numerical features.

We created specific features to capture hidden value in the data:

**A. The "Staleness" Metric (`days_since_review`)**
We hypothesized that listings with outdated reviews are less desirable. We converted the `last_review` date into a numerical value representing the days elapsed since the last activity.

**B. Spatial Encoding**
Instead of using specific neighborhood names (which would create 200+ unnecessary columns), we utilized **One-Hot Encoding** for boroughs and relied on specific `latitude` and `longitude` coordinates for precision.

**Code Snippet:**
```python
# Converting date text into a measurable number
df['days_since_review'] = (latest_date - df['last_review']).dt.days
```

### Step 4: Model Training (Random Forest)
**Objective:** To train an algorithm that learns the non-linear relationship between features and price.

We selected the **Random Forest Regressor** as the primary model for this submission.

**Justification:**
Unlike Linear Regression, which assumes a straight-line relationship, a Random Forest builds 100 "Decision Trees" and averages their outputs. This allows the model to handle complex spatial patterns (e.g., pricing hotspots) without overfitting.

**Code Snippet:**
```python
from sklearn.ensemble import RandomForestRegressor

# Building 100 decision trees for stability
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
```

---

## 3. Results and Evaluation

The model was evaluated on a test set (20% of data) that it had never seen during training to ensure unbiased results.

**Key Findings:**
* **R² Score (Accuracy):** **0.54**
* **Mean Absolute Error (MAE):** **~$37.26**

**Interpretation:**
The model explains approximately 54% of the variance in listing prices. On average, the predicted price is within **$37** of the actual listing price. This indicates that while the model is effective at estimating a "ballpark" figure, it lacks the precision required for high-stakes valuation.

*(Note: As mentioned in the version history, subsequent experiments with XGBoost (v3.0) raised this score to ~0.61, highlighting the potential for further optimization.)*

---

## 4. Limitations

It is important to acknowledge the constraints of this study:

1.  **Missing Physical Attributes:** The dataset does not include **Square Footage**. In real estate, size is a critical factor. Without this variable, the model cannot distinguish between a small room and a large suite in the same location, which limits accuracy to the observed ~54%.
2.  **Condition Variables:** There is no data regarding the interior condition (renovated vs. old).
3.  **Geographic Scope:** The model is specific to NYC and cannot be generalized to other cities.

---

## 5. Conclusion

This project successfully demonstrated the implementation of an end-to-end machine learning pipeline. By processing raw data and applying domain-specific feature engineering (such as the Staleness Metric), we built a functional predictive tool.

While the accuracy is capped by the absence of property size data, the **Random Forest** model proved effective at capturing spatial pricing trends. This system serves as a robust proof-of-concept for an automated valuation tool, providing a solid foundation for future enhancements.

---
---

## Appendix: Project Context

Hey there, Louis here!

This repository is a result of a class assigment/project studies, issued by Mr Harry Atieku-Boateng - [@hatieku-boateng](https://github.com/hatieku-boateng),
Pentecost University, for the level 400 IT class.
This aim to level up each and everyone's Data analytical skills.

Find below a brief explanation of the work:
Dear Students,

Please find attached a dataset that will be used as a practical learning resource for developing your skills in machine learning, data exploration, and data analytics.
This dataset has been intentionally provided in a raw and realistic form to reflect the kinds of data challenges you are likely to encounter in real-world analytical and research contexts.

**Dataset Overview**
The dataset contains a collection of structured records representing multiple variables of interest across several observations.
It includes a mixture of numerical and categorical features, allowing you to practise a wide range of analytical techniques, from basic descriptive statistics to more advanced predictive modelling.
Some variables may exhibit patterns, relationships, or trends that are not immediately obvious, encouraging careful exploration and interpretation.

**Learning Objectives**
This dataset is provided to help you:

* Perform exploratory data analysis (EDA), including summary statistics, visualisation, and pattern discovery.
* Identify data quality issues such as missing values, inconsistencies, or outliers, and apply appropriate data cleaning techniques.
* Engineer and transform features where necessary to improve analytical or model performance.
* Apply suitable machine learning algorithms, whether for prediction, classification, or clustering, depending on how you frame your analytical task.
* Evaluate model performance using appropriate metrics and justify your methodological choices.

**Expectations and Approach**
You are encouraged to approach this dataset analytically and critically. There is no single “correct” outcome. Instead, emphasis should be placed on your reasoning process, the appropriateness of your techniques, and your ability to clearly explain insights derived from the data. Assumptions made during analysis should be explicitly stated and justified.

**Academic Integrity**
While you may discuss general ideas and approaches with peers, all analysis, code, and interpretations submitted must be your own work. Any external tools, libraries, or references used should be properly acknowledged.

Kind regards,
Harry Atieku-Boateng
