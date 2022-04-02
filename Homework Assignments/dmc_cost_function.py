import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix

def cost_function(Y_true, Y_preds):
    
    '''
    This a customized scoring function that takes two arguments:
    Y_true: true labels
    Y_preds: likelihoods from the model   
    '''
    
    ## Defining cutoff values in a data-frame
    results = pd.DataFrame({'cutoffs': np.round(np.linspace(0.05, 0.95, num = 40, endpoint = True), 2)})
    results['cost'] = np.nan
    
    for i in range(0, results.shape[0]):
        
        ## Changing likelihoods to labels
        Y_label = np.where(Y_preds < results.at[i, 'cutoffs'], 0, 1)
        
        ## Computing confusion matrix and scoring based on description
        X = confusion_matrix(Y_label, Y_true)
        results.at[i, 'cost'] = (0 * X[0, 0]) - (25 * X[1, 0]) - (5 * X[0, 1]) + (5 * X[1, 1])
        
    ## Sorting results 
    results = results.sort_values(by = 'cost', ascending = False).reset_index(drop = True)
    
    return results.at[0, 'cost']


def cost_function_cutoff(Y_true, Y_pred):
    
    ## Defining cutoff values in a data-frame
    results = pd.DataFrame({'cutoffs': np.round(np.linspace(0.05, 0.95, num = 40, endpoint = True), 2)})
    results['cost'] = np.nan
    
    for i in range(0, results.shape[0]):
        
        ## Changing likelihoods to labels
        Y_pred_lab = np.where(Y_pred < results.at[i, 'cutoffs'], 0, 1)
        
        ## Computing confusion matrix and scoring based on description
        X = confusion_matrix(Y_pred_lab, Y_true)
        results.at[i, 'cost'] = (0 * X[0, 0]) - (25 * X[1, 0]) - (5 * X[0, 1]) + (5 * X[1, 1])
        
    ## Sorting results 
    results = results.sort_values(by = 'cost', ascending = False).reset_index(drop = True)
    
    return results.at[0, 'cutoffs']


