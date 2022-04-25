'''
    Feature Selection:
    Using the RFECV algorithm with multiple
    model types and extracting results as csv files
'''

## Importing libraries
import boto3
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import RFECV
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
pd.set_option('display.max_columns', 100)

## --------------------------------------

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


## --------------------------------------

## Variable Engineering:

## BMI Categoricals
train['BMI_Underweight'] = np.where(train['BMI'] < 18.5, 1, 0)
train['BMI_Healthy'] = np.where((train['BMI'] >= 18.5) & (train['BMI'] < 25), 1, 0)
train['BMI_Overweight'] = np.where((train['BMI'] >= 25) & (train['BMI'] < 30), 1, 0)
train['BMI_Obese'] = np.where(train['BMI'] >= 30, 1, 0)

## Log(BMI)
train['Log_BMI'] = np.log(train['BMI'])

## Creating dummy variables for Education, and Income

train = pd.concat([train.drop(columns = ['Education']), pd.get_dummies(train['Education'])], axis = 1)
train = train.rename(columns = { 1: 'Never_Attended', 2: 'Grades_1_8', 3: 'Grades_9_11', 4: 'GED', 5: 'College_1_3', 
                              6: 'College_4+'})

train = pd.concat([train.drop(columns = ['Income']), pd.get_dummies(train['Income'])], axis = 1)
train = train.rename(columns = { 1: '<10,000', 2: '<15,000', 3: '<20,000', 4: '<25,000', 5: '<35,000', 
                                      6: '<50,000',  7: '<75,000',  8: '75,000+'})

## Other
train['MentHlth_cat'] = np.where((train.MentHlth <=10), 0, 
                                 np.where((train.MentHlth > 10) & (train.MentHlth <= 20), 1, 2))

train['PhysHlth_cat'] = np.where((train.PhysHlth <=10), 0, 
                              np.where((train.PhysHlth > 10) & (train.PhysHlth <= 20), 1, 2))

train['GenHlth_cat'] = np.where((train.GenHlth <=2), 0, 
                             np.where((train.GenHlth > 3) & (train.GenHlth <= 5), 1, 2))

## Creating interactions of top variables
train['Interaction_1'] = train['HighBP'] * train['GenHlth']
train['Interaction_2'] = train['HighBP'] * train['GenHlth_cat']
train['Interaction_3'] = train['HighBP'] * train['HighChol']
train['Interaction_4'] = train['GenHlth'] * train['GenHlth_cat']
train['Interaction_5'] = train['GenHlth'] * train['HighChol']
train['Interaction_6'] = train['GenHlth_cat'] * train['HighChol']

## Creating tree interactions
train['Tree_1'] = np.where((train['Interaction_2'] <= 0.5) & (train['Interaction_5'] <= 1.5) & (train['Age'] <= 8.5), 1, 0)
train['Tree_2'] = np.where((train['Interaction_2'] <= 0.5) & (train['Interaction_5'] <= 1.5) & (train['Age'] > 8.5), 1, 0)
train['Tree_3'] = np.where((train['Interaction_2'] <= 0.5) & (train['Interaction_5'] > 1.5) & (train['Log_BMI'] <= 3.384), 1, 0)
train['Tree_4'] = np.where((train['Interaction_2'] <= 0.5) & (train['Interaction_5'] > 1.5) & (train['Log_BMI'] > 3.384), 1, 0)
train['Tree_5'] = np.where((train['Interaction_2'] > 0.5) & (train['Interaction_5'] <= 3.5) & (train['BMI'] <= 30.5), 1, 0)
train['Tree_6'] = np.where((train['Interaction_2'] > 0.5) & (train['Interaction_5'] <= 3.5) & (train['BMI'] > 30.5), 1, 0)
train['Tree_7'] = np.where((train['Interaction_2'] > 0.5) & (train['Interaction_5'] > 3.5) & (train['Log_BMI'] <= 3.481), 1, 0)
train['Tree_8'] = np.where((train['Interaction_2'] > 0.5) & (train['Interaction_5'] > 3.5) & (train['Log_BMI'] > 3.481), 1, 0)

## Printing the first five observations
print(train.head())
print(train.shape)


## --------------------------------------

## Defining the input and target variables
X = train.drop(columns = ['Diabetes_012'])
Y = train['Diabetes_012']

## Splitting the data into training, validation, and testing sets
X_training, X_validation, Y_training, Y_validation = train_test_split(X, Y, test_size = 0.2, stratify = Y)

## --------------------------------------

## Defining all functions

#######################################
## RFECV with DecisionTreeClassifier ##
#######################################

def decisionTree(X_training, Y_training):
    
    ## Defining empty lists to store results
    variable_support = []
    
    for i in tqdm(range(0, 10)):

        ## Defining the binary Y data for the class 0
        Y_training_training = np.where(Y_training == 0, 1, 0)

        ## Building the RFECV model
        tree_rfecv = RFECV(estimator = DecisionTreeClassifier(), step = 1, min_features_to_select = 2, 
                           cv = 3, scoring = 'f1', n_jobs = -1).fit(X_training, Y_training_training)

        ## Appending results to list
        variable_support.append(tree_rfecv.support_)

        ## ----------------

        ## Defining the binary Y data for the class 1
        Y_training_training = np.where(Y_training == 1, 1, 0)

        ## Building the RFECV model
        tree_rfecv = RFECV(estimator = DecisionTreeClassifier(), step = 1, min_features_to_select = 2, 
                           cv = 3, scoring = 'f1', n_jobs = -1).fit(X_training, Y_training_training)

        ## Appending results to list
        variable_support.append(tree_rfecv.support_)

        ## ----------------

        ## Defining the binary Y data for the class 2
        Y_training_training = np.where(Y_training == 2, 1, 0)

        ## Building the RFECV model
        tree_rfecv = RFECV(estimator = DecisionTreeClassifier(), step = 1, min_features_to_select = 2, 
                           cv = 3, scoring = 'f1', n_jobs = -1).fit(X_training, Y_training_training)

        ## Appending results to list
        variable_support.append(tree_rfecv.support_)
        
    ## Extracting variable selection results
    support = pd.DataFrame(variable_support, columns = X_training.columns)
    support_2 = 100 * support.apply(np.sum, axis = 0) / support.shape[0]
    support_3 = pd.DataFrame({'Variable': support_2.index, 'Score': support_2.values})
    
    ## Exporting results as a csv file
    support_3.to_csv('DTC_RFECV.csv', index = False)
    
    
## --------------------------------------

    
#######################################
## RFECV with RandomForestClassifier ##
#######################################

def randomForest(X_training, Y_training):
    
    ## Defining empty lists to store results
    variable_support = []
    
    for i in tqdm(range(0, 10)):

        ## Defining the binary Y data for the class 0
        Y_training_training = np.where(Y_training == 0, 1, 0)

        ## Building the RFECV model
        rf_rfecv = RFECV(estimator = RandomForestClassifier(), step = 1, min_features_to_select = 2, 
                           cv = 3, scoring = 'f1', n_jobs = -1).fit(X_training, Y_training_training)

        ## Appending results to list
        variable_support.append(rf_rfecv.support_)

        ## ----------------

        ## Defining the binary Y data for the class 1
        Y_training_training = np.where(Y_training == 1, 1, 0)

        ## Building the RFECV model
        rf_rfecv = RFECV(estimator = RandomForestClassifier(), step = 1, min_features_to_select = 2, 
                           cv = 3, scoring = 'f1', n_jobs = -1).fit(X_training, Y_training_training)

        ## Appending results to list
        variable_support.append(rf_rfecv.support_)

        ## ----------------

        ## Defining the binary Y data for the class 2
        Y_training_training = np.where(Y_training == 2, 1, 0)

        ## Building the RFECV model
        rf_rfecv = RFECV(estimator = RandomForestClassifier(), step = 1, min_features_to_select = 2, 
                           cv = 3, scoring = 'f1', n_jobs = -1).fit(X_training, Y_training_training)

        ## Appending results to list
        variable_support.append(rf_rfecv.support_)
        
    ## Extracting variable selection results
    support = pd.DataFrame(variable_support, columns = X_training.columns)
    support2 = 100 * support.apply(np.sum, axis = 0) / support.shape[0]
    support3 = pd.DataFrame({'Variable': support2.index, 'Score': support2.values})
    
    ## Exporting results as a csv file
    support3.to_csv('RF_RFECV.csv', index = False)
    

## --------------------------------------

###################################################################
## RFECV with AdaBoostClassifier w/ DecisionTreeClassifier base ###
###################################################################

def adaBoost(X_training, Y_training):
    
    ## Defining empty lists to store results
    variable_support = []
    
    for i in tqdm(range(0, 10)):

        ## Defining the binary Y data for the class 0
        Y_training_training = np.where(Y_training == 0, 1, 0)

        ## Building the RFECV model
        ada_rfecv = RFECV(estimator = AdaBoostClassifier(), step = 1, min_features_to_select = 2, 
                           cv = 3, scoring = 'f1', n_jobs = -1).fit(X_training, Y_training_training)

        ## Appending results to list
        variable_support.append(ada_rfecv.support_)

        ## ----------------

        ## Defining the binary Y data for the class 1
        Y_training_training = np.where(Y_training == 1, 1, 0)

        ## Building the RFECV model
        ada_rfecv = RFECV(estimator = AdaBoostClassifier(), step = 1, min_features_to_select = 2, 
                           cv = 3, scoring = 'f1', n_jobs = -1).fit(X_training, Y_training_training)

        ## Appending results to list
        variable_support.append(ada_rfecv.support_)

        ## ----------------

        ## Defining the binary Y data for the class 2
        Y_training_training = np.where(Y_training == 2, 1, 0)

        ## Building the RFECV model
        ada_rfecv = RFECV(estimator = AdaBoostClassifier(), step = 1, min_features_to_select = 2, 
                           cv = 3, scoring = 'f1', n_jobs = -1).fit(X_training, Y_training_training)

        ## Appending results to list
        variable_support.append(ada_rfecv.support_)
        
    ## Extracting variable selection results
    support = pd.DataFrame(variable_support, columns = X_training.columns)
    support2 = 100 * support.apply(np.sum, axis = 0) / support.shape[0]
    support3 = pd.DataFrame({'Variable': support2.index, 'Score': support2.values})
    
    ## Exporting results as a csv file
    support3.to_csv('ADA_RFECV.csv', index = False)
    

## --------------------------------------

###########################################
## RFECV with GradientBoostingClassifier ##
###########################################

def gradientBoost(X_training, Y_training):
    
    ## Defining empty lists to store results
    variable_support = []
    
    for i in tqdm(range(0, 10)):

        ## Defining the binary Y data for the class 0
        Y_training_training = np.where(Y_training == 0, 1, 0)

        ## Building the RFECV model
        gb_rfecv = RFECV(estimator = GradientBoostingClassifier(), step = 1, min_features_to_select = 2, 
                           cv = 3, scoring = 'f1', n_jobs = -1).fit(X_training, Y_training_training)

        ## Appending results to list
        variable_support.append(gb_rfecv.support_)

        ## ----------------

        ## Defining the binary Y data for the class 1
        Y_training_training = np.where(Y_training == 1, 1, 0)

        ## Building the RFECV model
        gb_rfecv = RFECV(estimator = GradientBoostingClassifier(), step = 1, min_features_to_select = 2, 
                           cv = 3, scoring = 'f1', n_jobs = -1).fit(X_training, Y_training_training)

        ## Appending results to list
        variable_support.append(gb_rfecv.support_)

        ## ----------------

        ## Defining the binary Y data for the class 2
        Y_training_training = np.where(Y_training == 2, 1, 0)

        ## Building the RFECV model
        gb_rfecv = RFECV(estimator = GradientBoostingClassifier(), step = 1, min_features_to_select = 2, 
                           cv = 3, scoring = 'f1', n_jobs = -1).fit(X_training, Y_training_training)

        ## Appending results to list
        variable_support.append(gb_rfecv.support_)
        
    ## Extracting variable selection results
    support = pd.DataFrame(variable_support, columns = X_training.columns)
    support2 = 100 * support.apply(np.sum, axis = 0) / support.shape[0]
    support3 = pd.DataFrame({'Variable': support2.index, 'Score': support2.values})
    
    ## Exporting results as a csv file
    support3.to_csv('GB_RFECV.csv', index = False)
    
    

## --------------------------------------
    
    
# Calling functions
print('\n-- Beginning: RFECV with Decision Tree Classifier --\n')
decisionTree(X_training, Y_training)

print('\n-- Beginning: RFECV with Random Forest Classifier --\n')
randomForest(X_training, Y_training)

print('\n-- Beginning: RFECV with AdaBoost Classifier --\n')
adaBoost(X_training, Y_training)

print('\n-- Beginning: RFECV with Gradient Boosting Classifier --\n')
gradientBoost(X_training, Y_training)
