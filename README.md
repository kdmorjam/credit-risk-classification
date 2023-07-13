# credit-risk-classification

## Overview of the Analysis

This predictive analysis is to determine whether the future outcome of a loan would be at risk or not.  The metrics (features) 
used in this analysis was loan size, interest rate, income of borrower, debt to income ratio, number of loan accounts, 
derogatory mark, and total of all debts.

Supervised Machine Learning model with Logistic regression was used as there was a binary outcome (loan_status with 1 or 0) 
based on the linear relationship between the features mentioned previously. The machine learning process was as follows:
* Read lending CSV into a DataFrame, before separating data into features (X) and target (y). The target was converted to a 
  numpy array (values.reshape[-1,1])
* Checked the distribution of target values using np.unique(). At this stage it was observeded that the data was imbalanced as follows:
  [    0    75036]
  [    1     2500]
* The data was then split 70:30, into traininig (X_train, y_train) and testing(X_test, y_test) datasets using train_test_split
* The training data (X_train & y_train) was fitted to the LogisticRegression() model
* The testing data, X_test was used to predict the outcome of the model.
* The model was then evaluated by determining the balanced accuracy score, confusion matrix and classification report. 
  Balanced accuracy was determined instead of accuracy, as the dataset was imbalanced.

A second LogisticRegression model was created using resampled training data. The oversampling strategy was used by calling the 
RandomOverSampler() function to "balance" the training dataset. The oversampling resulted in a distribution as follows:
  [    0  52568]
  [    1  52568]

* The model was then evaluated by determining the balanced accuracy score, confusion matrix and classification report.


## Results

* Machine Learning Model 1 - Imbalanced dataset:

  * The balanced accuracy was determined to be: 0.9481269535981227
  * The precision for healthy loans was 1 which indicated all predictions of healthy loans was correct. However, for high-risk loans,
    the precision was 0.86 (86%) of total positive observations.
  * Recall was 0.99 for healthy loans and 0.90 for high-risk. Healthy loans were therfore classified correctly 99% of the time by the 
    model and high-risk loans at 90%. 

          confusion matrix
                    Predicted 0	  Predicted 1
          Actual 0	      22347	          121
          Actual 1	         78	          715


          classification report
                          precision    recall  f1-score   support

            Healthy loan       1.00      0.99      1.00     22468
          High-risk loan       0.86      0.90      0.88       793

                accuracy                           0.99     23261
               macro avg       0.93      0.95      0.94     23261
            weighted avg       0.99      0.99      0.99     23261


* Machine Learning Model 2 - Resample:

  * The balanced accuracy was determined to be: 0.9931903712406109
  * The precision for healthy loans was 1 which indicated all predictions of healthy loans was correct. However, for high-risk loans,
    the precision was 0.85 (85%) of total positive observations.
  * Recall was 0.99 for both healthy and high-risk loans, indicating 99% of the time the model classified both loan types correctly. 
  
          confusion matrix
                    Predicted 0	  Predicted 1
          Actual 0	      22389	          131
          Actual 1	          2	          739


          classification report
                          precision    recall  f1-score   support

            Healthy loan       1.00      0.99      1.00     22468
          High-risk loan       0.85      0.99      0.92       793

                accuracy                           0.99     23261
               macro avg       0.93      0.99      0.96     23261
            weighted avg       0.99      0.99      0.99     23261

## Summary

Based on the results, it would be recommended to use the second model. This model had a balanced accuracy 99.3% whereas, the first 
model's accuracy was 94.8%. Additionally, both models had the same precision, recall and f1-core for healthy loans. However, for high-risk loans 
recall was 99% in model 2 versus 90% in models 1. The F1-score was 92% versus 88% in model 1. Overall, model 2 was better at determining whether 
or not a future loan would be at-risk. 
