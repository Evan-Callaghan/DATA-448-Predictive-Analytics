'''
    Feature Selection:
    Using the RFECV algorithm with multiple
    model types and extracting results
'''

## Importing libraries
import boto3
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_selection import RFECV
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
pd.set_option('display.max_columns', 50)

## Using pandas to read the training and testing data files
## Defining the bucket
s3 = boto3.resource('s3')
bucket_name = 'data-448-bucket-callaghan'
bucket = s3.Bucket(bucket_name)

file_key = 'diabetes_train.csv'
file_key2 = 'diabetes_test.csv'

bucket_object = bucket.Object(file_key)
bucket_object2 = bucket.Object(file_key2)

file_object = bucket_object.get()
file_object2 = bucket_object2.get()

file_content_stream = file_object.get('Body')
file_content_stream2 = file_object2.get('Body')

train = pd.read_csv(file_content_stream)
test = pd.read_csv(file_content_stream2)

print(train.head())


## Calling functions

decisionTree(X_training, Y_training)


## Defining all functions

## --------------------------------------

## RFECV with DecisionTreeClassifier

def decisionTree(X_training, Y_training):
    
    for i in range(0, 20):

        ## Defining empty lists to store results
        variable_support = []

        ## Defining the binary Y data for the class 0
        Y_training_training = np.where(Y_training == 0, 1, 0)

        ## Building the RFECV model
        tree_rfecv = RFECV(estimator = DecisionTreeClassifier(), step = 1, min_features_to_select = 2, 
                           cv = 3, scoring = 'f1').fit(X_training, Y_training_training)

        ## Appending results to list
        variable_support.append(tree_rfecv.support_)

        ## ----------------

        ## Defining the binary Y data for the class 1
        Y_training_training = np.where(Y_training == 1, 1, 0)

        ## Building the RFECV model
        tree_rfecv = RFECV(estimator = DecisionTreeClassifier(), step = 1, min_features_to_select = 2, 
                           cv = 3, scoring = 'f1').fit(X_training, Y_training_training)

        ## Appending results to list
        variable_support.append(tree_rfecv.support_)

        ## ----------------

        ## Defining the binary Y data for the class 2
        Y_training_training = np.where(Y_training == 2, 1, 0)

        ## Building the RFECV model
        tree_rfecv = RFECV(estimator = DecisionTreeClassifier(), step = 1, min_features_to_select = 2, 
                           cv = 3, scoring = 'f1').fit(X_training, Y_training_training)

        ## Appending results to list
        variable_support.append(tree_rfecv.support_)
        
    ## Extracting variable selection results
    support = pd.DataFrame(variable_support, columns = X_training.columns)
    support = 100 * support.apply(np.sum, axis = 0) / support.shape[0]
    
    ## Exporting results as a csv file
    support.to_csv('DTC_RFECV.csv', index = False)
    
    
    ## --------------------------------------

## RFECV with RandomForestClassifier

def randomForest(X_training, Y_training):
    
    for i in range(0, 20):

        ## Defining empty lists to store results
        variable_support = []

        ## Defining the binary Y data for the class 0
        Y_training_training = np.where(Y_training == 0, 1, 0)

        ## Building the RFECV model
        rf_rfecv = RFECV(estimator = RandomForestClassifier(), step = 1, min_features_to_select = 2, 
                           cv = 3, scoring = 'f1').fit(X_training, Y_training_training)

        ## Appending results to list
        variable_support.append(rf_rfecv.support_)

        ## ----------------

        ## Defining the binary Y data for the class 1
        Y_training_training = np.where(Y_training == 1, 1, 0)

        ## Building the RFECV model
        rf_rfecv = RFECV(estimator = RandomForestClassifier(), step = 1, min_features_to_select = 2, 
                           cv = 3, scoring = 'f1').fit(X_training, Y_training_training)

        ## Appending results to list
        variable_support.append(rf_rfecv.support_)

        ## ----------------

        ## Defining the binary Y data for the class 2
        Y_training_training = np.where(Y_training == 2, 1, 0)

        ## Building the RFECV model
        rf_rfecv = RFECV(estimator = RandomForestClassifier(), step = 1, min_features_to_select = 2, 
                           cv = 3, scoring = 'f1').fit(X_training, Y_training_training)

        ## Appending results to list
        variable_support.append(rf_rfecv.support_)
        
    ## Extracting variable selection results
    support = pd.DataFrame(variable_support, columns = X_training.columns)
    support = 100 * support.apply(np.sum, axis = 0) / support.shape[0]
    
    ## Exporting results as a csv file
    support.to_csv('RF_RFECV.csv', index = False)
    

## --------------------------------------

## RFECV with AdaBoostClassifier w/ DecisionTreeClassifier base

def adaBoost(X_training, Y_training):
    
    for i in range(0, 20):

        ## Defining empty lists to store results
        variable_support = []

        ## Defining the binary Y data for the class 0
        Y_training_training = np.where(Y_training == 0, 1, 0)

        ## Building the RFECV model
        ada_rfecv = RFECV(estimator = AdaBoostClassifier(base_estimator = DecisionTreeClassifier()), step = 1, min_features_to_select = 2, 
                           cv = 3, scoring = 'f1').fit(X_training, Y_training_training)

        ## Appending results to list
        variable_support.append(ada_rfecv.support_)

        ## ----------------

        ## Defining the binary Y data for the class 1
        Y_training_training = np.where(Y_training == 1, 1, 0)

        ## Building the RFECV model
        ada_rfecv = RFECV(estimator = AdaBoostClassifier(base_estimator = DecisionTreeClassifier()), step = 1, min_features_to_select = 2, 
                           cv = 3, scoring = 'f1').fit(X_training, Y_training_training)

        ## Appending results to list
        variable_support.append(ada_rfecv.support_)

        ## ----------------

        ## Defining the binary Y data for the class 2
        Y_training_training = np.where(Y_training == 2, 1, 0)

        ## Building the RFECV model
        ada_rfecv = RFECV(estimator = AdaBoostClassifier(base_estimator = DecisionTreeClassifier()), step = 1, min_features_to_select = 2, 
                           cv = 3, scoring = 'f1').fit(X_training, Y_training_training)

        ## Appending results to list
        variable_support.append(ada_rfecv.support_)
        
    ## Extracting variable selection results
    support = pd.DataFrame(variable_support, columns = X_training.columns)
    support = 100 * support.apply(np.sum, axis = 0) / support.shape[0]
    
    ## Exporting results as a csv file
    support.to_csv('ADA_RFECV.csv', index = False)