# Credit_Risk_Analysis
## Overview
Supervised machine learning is a form of predictive analytics in which we choose the dependent variable for analysis. There are two main regression; Linear Regression are used to predict continuous variables while Logistic are used for predicting booleans. A linear regression is of the form dependent_variable = (const_1 * feat_1)+ ... +(const_n * feat_n); the predicted dependent variable is a linear combination of its features. A logistic variable is of the form log(dependent_variable/dependent_variable')= (const_1 * feat_1)+ ... +(const_n * feat_n); This relationship derives from the logit-normal pdf [Logistic Proof](https://www.countbayesie.com/blog/2021/9/30/the-logit-normal-a-ubitiqutious-but-strange-distribution). The truth value of a dependent logistic variable is determined by where it sits on the logistic curve; The graph splits results into left of the curve and right of the curve (or true and false).

### Purpose
Using Pythons' imbalanced-learn and sklearn predict credit risk 

## Analysis
Like most datasets, the distribution of our dependent variable is skewed since there are substancially less high-risk users than low-risk.

### Oversampling
Oversampling handles this problem by adding more datapoints to the under represented group; 
#### Naive Random 
#### SMOTE

### Underampling
Underampling takes datapoints from the over represented group; 
#### ClusterCentroids

### Combination Sampling
Combination sampling only keeps important datapoint, eliminating outliers to produce
#### SMOTEEN
#### Balanced Random Forest Classifier
#### EasyEnsembleClassifier
 
## Summary
