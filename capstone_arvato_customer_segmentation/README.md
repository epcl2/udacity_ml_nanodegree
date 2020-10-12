# Machine Learning Engineer Nanodegree Capstone Project

## Arvato: Customer Segmentation and Prediction

### Overview
This project is one of the suggested capstone projects for the Udacity ML Engineer Nanedegree. The datasets are made available by Arvato. 

I selected this project as it involves a real-life dataset, which not only is rather messy, but also has a lot of rows giving me the chance to improve my data processing skills. I then have the chance to test out my skills in the latter portion of this project by submitting results on a Kaggle competition.

The project consists of 2 major parts: customer segmentation and a supervised learning model to predict responses to a campaign by a mail-order company.

1. Customer segmentation
Unsupervised learning techniques (PCA, KMeans clustering) are used to segment the general population into clusters. Existing customers of a mail-order company are investigated to see how they fit in to the general population clusters. The clusters are then analysed to see what are the core characteristics of customers for the mail-order company.

2. Supervised Learning Model
Several supervised learning models are built to predict whether individuals will respond to a marketing campaign. The results of the model is then submitted to a Kaggle competition for evaluation.

### Dependencies
- pandas
- numpy 
- matplotlib
- seaborn
- scikit-learn
- xgboost
- lightgbm

### Contents
- `arvato1_data_clean.ipynb`: contains the data pre-processing and feature engineering part of this project.
- `arvato2_customer_segmentation.ipynb`: unsupervised learning techniques for customer segmentation
- `arvato3_prediction.ipynb`: supervised learning model to predict responses of individuals to a campaign
- `helpers.py`: helper functions for data pre-processing and running supervised learning models
