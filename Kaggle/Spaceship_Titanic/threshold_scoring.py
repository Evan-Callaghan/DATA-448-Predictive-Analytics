import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve


## Function to return the estimated likelihoods and the optimal cutoff value
def auc_roc_cutoff(Y_true, Y_preds):
    
    ## Computing the ROC-Curve
    fpr, tpr, threshold = roc_curve(Y_true, Y_preds)
    
    ## Creating a data frame to store ROC-Curve results
    cutoffs = pd.DataFrame({'False_Positive': fpr, 'True_Positive': tpr, 'Cutoff': threshold})
    
    ## Calculating the Euclidean distance between each point and our optimal model (0,1)
    cutoffs['True_Positive_Minus_1'] = cutoffs['True_Positive'] - 1
    cutoffs['Distance'] = np.sqrt(cutoffs['False_Positive']**2 + cutoffs['True_Positive_Minus_1']**2)
    
    ## Sorting the data frame based on Euclidean distance
    cutoffs = cutoffs.sort_values('Distance', ascending = True).reset_index(drop = True)
    
    ## Defining the optimal cutoff value
    optimal_cutoff = round(cutoffs['Cutoff'][0], 3)
    
    ## Changing likelihoods to labels
    Y_preds_labels = np.where(Y_preds < optimal_cutoff, 0, 1)
    
    ## Returning the estimated labels
    return Y_preds_labels, optimal_cutoff