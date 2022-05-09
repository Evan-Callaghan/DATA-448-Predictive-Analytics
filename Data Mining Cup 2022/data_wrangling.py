import pandas as pd
import numpy as np
from tqdm import tqdm

def cleaning_category(data):
    
    ## Defining new column
    data['new_category'] = np.nan
    
    n = data.shape[0]
    
    for i in range(0, n):
        x = data['categories'][i]
        if (pd.isna(x)):
            continue
        else :
            x = x.replace('[', '')
            x = x.replace(']', '')
            data.loc[i, 'new_category'] = x
    
    ## Splitting category
    categories = data['new_category'].str.split(',', expand = True)
    m = categories.shape[1]
    categories_names = ['category_' + str(i) for i in range(1, (m+1))]
    categories.columns = categories_names
    categories = categories.fillna(value = np.nan)
    
    ## Appending split categories
    data = pd.concat([data, categories], axis = 1)
    data = data.drop(columns = ['categories', 'new_category'], axis = 1)
    
    return data


def appending_parent_category(orders_items, parent_category):
    
    ## Extracting categories from orders_items
    categories_names = ['category_' + str(i) for i in range(1, 36)]
    categories = orders_items[categories_names]
    
    ## Appending parent categories columns
    parent_names = ['parent_category_' + str(i) for i in range(1, 36)]
    
    for i in range(0, len(parent_names)):
        categories[parent_names[i]] = np.nan
    
    ## Looping through all rows of categories to append parent categories
    m = categories.shape[0]
    
    for i in tqdm(range(0, m)):
        temp_data = categories.loc[i].dropna() 
        temp_data = pd.DataFrame(temp_data).reset_index(drop = True)
        temp_data.columns = ['category']
        temp_data['category'] = pd.to_numeric(temp_data['category'])
        
        temp_data = pd.merge(temp_data, parent_category, on = 'category', how = 'left')
        to_append = temp_data['parent_category'].unique()
        
        n = to_append.shape[0]
        
        for j in range(35, (35+n)):
            categories.iloc[i, j] = to_append[j-35]
        
    return categories
    

def removing_NAN_col(data):
    
    ## Identifying columns filled with NAN & dropping them 
    to_remove = data.columns[data.isna().all()].to_list()
    data_out = data.drop(columns = to_remove, axis = 1)
    
    return data_out
    

    
def selecting_items_category_based(submission, orders_items, category, items):
    
    ## Defining list to store data 
    items_data = list()
    
    ## Selecting unique itemID
    submission_items = submission['itemID'].unique()
    
    for i in tqdm(range(0, len(submission_items))):
        
        temp_data = orders_items[orders_items['itemID'] == submission_items[i]].reset_index(drop = True)
        temp_data = removing_NAN_col(temp_data)
        
        ## Extracting parent categories
        parent_category = [i for i in temp_data.columns if 'parent_category_' in i]
        parent_category = np.unique(temp_data[parent_category].values)

        ## Extracting all categories associated to the above parent categories
        categories = category[np.isin(category['parent_category'], parent_category)]['category'].unique()
        
        ## Extracting and storing all items
        items_data.append(selecting_items_category_based_help(categories, items, submission_items[i]))
        
    return pd.concat(items_data)
        
        
    
def selecting_items_category_based_help(categories, items, submission_item):    
    
    ## Defining list to store results 
    items_results = list()
    
    for i in range(0, items.shape[0]):
        x = items['categories'][i]
        if (pd.isna(x)):
            continue
        else :
            x = x.replace('[', '')
            x = x.replace(']', '')
            x = x.split(',')
            x = list(map(int, x))
            check = np.isin(x, categories)
            if (sum(check) > 0):
                items_results.append(items['itemID'][i])
                
    ## Defining data-frame to be exported
    item_data_out = pd.DataFrame({'itemID': list(np.repeat(submission_item, len(items_results))), 'items': items_results})
    
    return item_data_out
    
    

def user_item_freq_buy(data, item):
    
    ## Subsetting the data 
    temp_data = data[data['itemID'] == item]
    
    ## Aggregrating data at user level
    data_out = pd.DataFrame(temp_data.groupby('userID')['order'].sum())
    data_out['userID'] = data_out.index
    
    time_diff = (temp_data['date'].tail(1).values - temp_data['date'].head(1).values) / np.timedelta64(1, 'W')
    
    data_out['tot_weeks'] = np.repeat(time_diff, data_out.shape[0])
    data_out['freq_buy'] = data_out['tot_weeks'] / data_out['order']
    
    return data_out[['userID', 'order', 'tot_weeks', 'freq_buy']].reset_index(drop = True)