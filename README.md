# Credit_Risk_Analysis
## Overview
Supervised machine learning is a form of predictive analytics in which we choose the dependent variable for analysis. There are two main regression; Linear Regression are used to predict continuous variables while Logistic is used for predicting booleans. A linear regression is of the form dependent_variable = (const_1 * feat_1)+ ... +(const_n * feat_n); the predicted dependent variable is a linear combination of its features. A logistic variable is of the form log(dependent_variable/dependent_variable')= (const_1 * feat_1)+ ... +(const_n * feat_n); This relationship derives from the logit-normal pdf [(Logistic Proof)](https://www.countbayesie.com/blog/2021/9/30/the-logit-normal-a-ubitiqutious-but-strange-distribution). The truth value of a dependent logistic variable is determined by where it sits on the logistic curve; The graph splits results into left of the curve and right of the curve (or true and false)

### Purpose
Using Pythons' imbalanced-learn and sklearn predict credit risk 

## Analysis
Like most datasets, the distribution of our dependent variable is skewed since there are substantially fewer high-risk users than low-risk

### Oversampling
Oversampling handles this problem by adding more data points to the underrepresented group <br /><br />
![Screen Shot 2022-07-24 at 9 34 44 AM](https://user-images.githubusercontent.com/79609464/180654669-a49ef372-ecc3-4120-ac69-cefd4aff205e.png)

##### Naive Random:
Minority class members are randomly selected and added to the training set until the majority and minority classes are balanced<br/>
![Screen Shot 2022-07-24 at 9 36 09 AM](https://user-images.githubusercontent.com/79609464/180654755-a81550b8-9f91-4a6e-9370-d9bebf66aec8.png)

##### SMOTE:
Or the Synthetic Minority Oversampling Technique is similar to native random, but instead of choosing random points to add to the minority class, the results are interpolated, adding only the closest neighbors and most relevant datapoint <br/>
![Screen Shot 2022-07-24 at 9 36 40 AM](https://user-images.githubusercontent.com/79609464/180654773-2270f479-61f2-4327-b297-8c7ea3b1b384.png)

- Though SMOTE is only choosing the most relevant points to expand our minority set, the accuracy is still poor, indicating Oversampling is not an accurate model
- How we oversample does not matter with these models, since their accuracy is within 1%
- Though we've added more data to the minority class, we haven't added the right data classified the right way, thus our predictions haven't changed much

### UnderSampling
Undersampling takes data from the overrepresented group <br /><br />
![Screen Shot 2022-07-24 at 9 37 05 AM](https://user-images.githubusercontent.com/79609464/180654804-dd9b3e16-6123-4218-b20a-39b37399f3e0.png)

##### Cluster Centroids:
Under samples the majority class by replacing a cluster of majority samples with a cluster centroid of the KMeans algorithm. The algorithm keeps N majority samples, fitting N cluster to the majority class and using the coordinates of the N cluster centroids as the new majority samples <br/>
![Screen Shot 2022-07-24 at 9 37 12 AM](https://user-images.githubusercontent.com/79609464/180654815-11e9d280-c2d9-4624-8b02-64ee17f3e4b5.png)

- The results are significantly worse than oversampling because we've accidentally eliminated too many key features of the model

### Combination Sampling
As the name implies, Combination Sampling is a combination of over and undersampling the data<br/><br/>
![Screen Shot 2022-07-24 at 9 38 31 AM](https://user-images.githubusercontent.com/79609464/180654870-3459ab5b-2a9d-4234-a379-f9d99453d2a2.png)

##### SMOTEENN:
This model is a two-step process. The first uses the normal SMOTE model to oversample the minority class. The second cleans data with undersampling; If the two nearest neighbors of a data point belong to two different classes, that data point is dropped- eliminating boundaries the model has trouble classifying <br/>
![Screen Shot 2022-07-24 at 9 39 58 AM](https://user-images.githubusercontent.com/79609464/180654940-3eb8897c-0ddc-47db-a952-ea80bb8dcc3c.png)

 - The results are more accurate than choosing solely over or under-sampling, but not by a large margin
 
### Ensemble Learners
Ensemble learners under sample subsetted clusters, then combine the results in an ensemble. Typically these clusters are bootstrapped, classified, and bagged into a new dataset

##### Balanced Random Forest Classifier:
The balanced random forest classifier randomly under samples each bootstrap sample to balance it.<br/>
![Screen Shot 2022-07-24 at 9 40 37 AM](https://user-images.githubusercontent.com/79609464/180654970-4129ebbe-dc5e-4c84-b125-a00655df01bf.png)
###### Top 15 Features
![Screen Shot 2022-07-24 at 9 41 07 AM](https://user-images.githubusercontent.com/79609464/180655004-0673771a-060a-4e6c-b2bd-631c5331b0f1.png)

##### Easy Ensemble Classifier:
An ensemble classified by AdaBoost learners, trained on different balanced bootstrap samples<br/>
![Screen Shot 2022-07-24 at 9 41 44 AM](https://user-images.githubusercontent.com/79609464/180655027-a97193a2-de01-4af5-a3cb-c825e333a5f9.png)
 
## Summary
The only model to produce remotely accurate results was the Easy Ensemble Classifier, every other model had roughly the same accuracy/precision of 60-70%. Undersampling is especially bad in this scenario because we are reducing an already 'small' sample of a couple thousand to a few hundred. Though ensemble learners also undersample, how they cluster and classify is completely different.

