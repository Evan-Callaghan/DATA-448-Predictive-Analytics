import pandas as pd
import numpy as np

def clean_method_mode(data, variable):
    
    ## Defining a list of indices where variable is missing
    item_index = data[data[variable].isnull()].index
    
    ## Extracting the mode from the variable of interest
    mode = data[variable].mode()[0]

    for item in item_index:

        ## If travelling individually, assigning mode as value
        if (data.at[item, 'GroupTotal'] == 1): 
            data.at[item, variable] = mode

        ## Otherwise, checking the number of people in travel party
        else:

            ## New data frame for members of the same party 
            new_data = data[(data['GroupNumber'] == data.at[item, 'GroupNumber']) & (data[variable].notnull())].reset_index(drop = True).head(1)

            ## If no valid people, assign mode
            if (new_data.shape[0] < 1):
                data.at[item, variable] = mode

            ## Else, assign value of group member
            else:
                data.at[item, variable] = new_data.at[0, variable]
                
    ## Returning revised data frame
    return data



def clean_method_mean(data, variable):
    
    ## Defining a list of indices where variable is missing
    item_index = data[data[variable].isnull()].index
    
    ## Extracting the mean from the variable of interest
    mean = data[variable].mean()

    for item in item_index:

        ## If travelling individually, assigning mean as value
        if (data.at[item, 'GroupTotal'] == 1): 
            data.at[item, variable] = mean

        ## Otherwise, checking the number of people in travel party
        else:

            ## New data frame for members of the same party
            new_data = data[(data['GroupNumber'] == data.at[item, 'GroupNumber']) & (data[variable].notnull())].reset_index(drop = True).head(1)

            ## If no valid people, assign mode
            if (new_data.shape[0] < 1):
                data.at[item, variable] = mean

            ## Else, assign value of group member
            else:
                data.at[item, variable] = new_data[variable].mean()
                
    ## Returning revised data frame
    return data



def clean_method_zero(data, variable):
    
    ## Defining a list of indices where variable is missing
    item_index = data[data[variable].isnull()].index

    for item in item_index:
        
        ## Assigning a value of zero for all missing values
        data.at[item, variable] = 0
                
    ## Returning revised data frame
    return data



def clean_method_string(data, variable):
    
    ## Defining a list of indices where variable is missing
    item_index = data[data[variable].isnull()].index

    for item in item_index:
        
        ## Assigning a value of zero for all missing values
        data.at[item, variable] = 'New String'
                
    ## Returning revised data frame
    return data



def clean_method_boolean(data, variable):
    
    ## Defining a list of indices where variable is missing
    item_index = data[data[variable].isnull()].index
    
    ## Extracting the mode from the variable of interest
    mode = data[variable].mode()[0]

    for item in item_index:

        ## If travelling individually, assigning mode as value
        if (data.at[item, 'GroupTotal'] == 1): 
            data.at[item, variable] = mode

        ## Otherwise, checking the number of people in travel party
        else:

            ## New data frame for members of the same party 
            new_data = data[(data['GroupNumber'] == data.at[item, 'GroupNumber']) & (data[variable].notnull())].reset_index(drop = True).head(1)

            ## If no valid people, assign mode
            if (new_data.shape[0] < 1):
                data.at[item, variable] = mode

            ## Else, assign value of group member
            else:
                data.at[item, variable] = new_data.at[0, variable]
                
    ## Returning revised data frame
    return data



def clean_method_cabin(data, variable):
    
    ## Defining a list of indices where variable is missing
    item_index = data[data[variable].isnull()].index

    for item in item_index:

        ## If travelling individually, assigning x/999/x as value
        if (data.at[item, 'GroupTotal'] == 1): 
            data.at[item, variable] = 'x/x/x'

        ## Otherwise, checking the number of people in travel party
        else:

            ## New data frame for members of the same party 
            new_data = data[(data['GroupNumber'] == data.at[item, 'GroupNumber']) & (data[variable].notnull())].reset_index(drop = True).head(1)

            ## If no valid people, assign mode
            if (new_data.shape[0] < 1):
                data.at[item, variable] = 'x/x/x'

            ## Else, assign value of group member
            else:
                data.at[item, variable] = new_data.at[0, variable]
                
    ## Returning revised data frame
    return data



def clean_method_cabin_2(data, variable):
    
    ## Defining a list of indices where variable is missing
    item_index = data[data[variable] == 'x'].index
    
    ## Extracting the mode from the variable of interest
    mode = data[data[variable] != 'x'][variable].mode()[0]

    for item in item_index:

        ## If travelling individually, assigning mode as value
        if (data.at[item, 'GroupTotal'] == 1): 
            data.at[item, variable] = mode

        ## Otherwise, checking the number of people in travel party
        else:

            ## New data frame for members of the same party 
            new_data = data[(data['GroupNumber'] == data.at[item, 'GroupNumber']) & (data[variable].notnull())].reset_index(drop = True).head(1)

            ## If no valid people, assign mode
            if (new_data.shape[0] < 1):
                data.at[item, variable] = mode

            ## Else, assign value of group member
            else:
                data.at[item, variable] = new_data.at[0, variable]
                
    ## Returning revised data frame
    return data