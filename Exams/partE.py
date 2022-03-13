import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier

## Reading the data-frame and printing the first five observations
training = pd.read_csv('training_data.csv')
print(training.head())

## Defining input and target variables
X_training = training.drop(columns = ['default payment next month'])
Y_training = training['default payment next month']

## Defining empty list to store results
rfe_results = []

## Repeating 100 times
for i in tqdm(range(0, 100)):
    
    ## Splitting the data
    X_training_train, X_training_test, Y_training_train, Y_training_test = train_test_split(X_training, Y_training, test_size = 0.2, stratify = Y_training)
    
    ## Running RFEC
    rf_rfecv = RFECV(estimator = RandomForestClassifier(max_depth = 3, n_estimators = 500), step = 1, 
                 min_features_to_select = 2, cv = 3).fit(X_training_train, Y_training_train)
    
    ## Appending features to be selected
    rfe_results.append(rf_rfecv.support_)
    print(i)
    

## Changing results list a to data-frame
rfe_results = pd.DataFrame(rfe_results, columns = X_training.columns)
results = 100 * rfe_results.apply(np.sum, axis = 0) / rfe_results.shape[0]

## Exporting the results data-frame as a csv for next steps in notebook file
results.to_csv('rfecv_results.csv', index = False)