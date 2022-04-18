import boto3
import pandas as pd; pd.set_option('display.max_columns', 50)
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier

## -----------------------------

## Using the pandas library to read the train.csv and test.csv data files and create two data-frames called train and test

## Defining the bucket
s3 = boto3.resource('s3')
bucket_name = 'data-448-bucket-callaghan'
bucket = s3.Bucket(bucket_name)

file_key = 'train(1).csv'
file_key2 = 'test(1).csv'

bucket_object = bucket.Object(file_key)
bucket_object2 = bucket.Object(file_key2)

file_object = bucket_object.get()
file_object2 = bucket_object2.get()

file_content_stream = file_object.get('Body')
file_content_stream2 = file_object2.get('Body')

train = pd.read_csv(file_content_stream)
test = pd.read_csv(file_content_stream2)

## -----------------------------

## Engineering features from Exam 1

## Train set:

## Most common repayment status
train['Most_Common'] = np.nan
for i in range(0, train.shape[0]):
    train.at[i, 'Most_Common'] = train[['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']].loc[i].mode()[0]

## From plot tree:
train['Tree2'] = np.where((train['PAY_0'] <= 1.5) & (train['PAY_2'] <= 1.5) & (train['PAY_AMT3'] > 395.0), 1, 0)
train['Tree6'] = np.where((train['PAY_0'] > 1.5) & (train['PAY_6'] <= 1.0) & (train['BILL_AMT1'] > 649.5), 1, 0)
train['Tree7'] = np.where((train['PAY_0'] > 1.5) & (train['PAY_6'] > 1.0) & (train['PAY_AMT3'] <= 14177.0), 1, 0)

## Test set:

## Most common repayment status
test['Most_Common'] = np.nan
for i in range(0, test.shape[0]):
    test.at[i, 'Most_Common'] = test[['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']].loc[i].mode()[0]

## From plot tree:
test['Tree2'] = np.where((test['PAY_0'] <= 1.5) & (test['PAY_2'] <= 1.5) & (test['PAY_AMT3'] > 395.0), 1, 0)
test['Tree6'] = np.where((test['PAY_0'] > 1.5) & (test['PAY_6'] <= 1.0) & (test['BILL_AMT1'] > 649.5), 1, 0)
test['Tree7'] = np.where((test['PAY_0'] > 1.5) & (test['PAY_6'] > 1.0) & (test['PAY_AMT3'] <= 14177.0), 1, 0)

## -----------------------------

## Splitting the train data-frame intro training (80%) and validation (20%) (taking into account the proportions of 0s and 1s)

## Defining the input and target variables
X = train.drop(columns = ['default payment next month'])
Y = train['default payment next month']

## Splitting the data
X_training, X_validation, Y_training, Y_validation = train_test_split(X, Y, test_size = 0.2, stratify = Y)

print('-- Starting in terminal --')

## -----------------------------

## Tuning the Random Forest model on the validation data-frame

## Defining the parameter dictionary
rf_param_grid = {'n_estimators': [100, 300, 500], 'max_depth': [3, 5, 7], 'min_samples_split': [5, 10, 15], 
                  'min_samples_leaf': [5, 10, 15]}

## Running GridSearchCV with 3 folds
rf_grid_search = GridSearchCV(RandomForestClassifier(), rf_param_grid, cv = 3, scoring = 'f1', n_jobs = -1).fit(X_validation, Y_validation)

## Extracting the best hyper-parameters
print('Optimal hyper-parameters for Random Forest Model: \n', rf_grid_search.best_params_)
print('\nOptimal F1-Score:\n', round(rf_grid_search.best_score_ * 100, 2), '%')

## -----------------------------

## Tuning the AdaBoost model on the validation data-frame

## Defining the parameter dictionary
ada_param_grid = {'n_estimators': [100, 300, 500], 'base_estimator__min_samples_split': [10, 15], 
                  'base_estimator__min_samples_leaf': [10, 15], 'base_estimator__max_depth': [3, 5, 7], 
                  'learning_rate': [0.001, 0.01, 0.1]}

## Running GridSearchCV with 3 folds
ada_grid_search = GridSearchCV(AdaBoostClassifier(base_estimator = DecisionTreeClassifier()), ada_param_grid, 
                               cv = 3, scoring = 'f1', n_jobs = -1).fit(X_validation, Y_validation)

## Extracting the best hyper-parameters
print('Optimal hyper-parameters for AdaBoost Model: \n', ada_grid_search.best_params_)
print('\nOptimal F1-Score:\n', round(ada_grid_search.best_score_ * 100, 2), '%')

## -----------------------------

## Tuning the XGBoost model on the validation data-frame

## Defining the parameter dictionary
xgb_param_grid = {'n_estimators': [300, 500], 'max_depth': [5, 7], 'min_child_weight': [5, 7], 
                  'learning_rate' : [0.01, 0.001], 'gamma': [0.1, 0.01], 'subsample': [0.8, 1], 
                  'colsample_bytree': [0.8, 1], 'early_stopping_rounds': [100]}

## Running GridSearchCV with 3 folds
xgb_grid_search = GridSearchCV(XGBClassifier(eval_metric = 'error', use_label_encoder = False), xgb_param_grid, cv = 3, scoring = 'f1', 
                               n_jobs = -1).fit(X_validation, Y_validation)

## Extracting the best hyper-parameters
print('Optimal hyper-parameters for XGBoost Model: \n', xgb_grid_search.best_params_)
print('\nOptimal F1-Score:\n', round(xgb_grid_search.best_score_ * 100, 2), '%')