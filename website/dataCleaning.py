import pandas as pd
import numpy as np
from PIL import Image
import PIL
import os
import time
import re

numerical_type=['int64','float64']
categorical_type=['object']

def cleanData(df, fname, dict, modelType, x):

    df_cat = pd.DataFrame()
    df_num = pd.DataFrame()
    df_index = pd.DataFrame()
    df_index_initial = pd.DataFrame()
    df_date = pd.DataFrame()

    col_dict = {
        'original':[],
        'mean':[],
        'median':[]
    }

    df_dict = {
        'index': df_index,
        'numerical': df_num,
        'categorical': df_cat,
        'date': df_date,
        'target': dict['target']
    }

    cleanMethod_dict = {

    }

    for col_name in df:

        cleanMethod_dict[col_name] = {
            'method' : [],
            'flag' : 0,
            'problem' : []
        }

    new_filename = 0

    verifyFile = True

    # Verify Target column suited modelType or not

    target = df_dict['target']
    y = df[target]

    y_drop_row = []

    if y.nunique() <= 1:
        verifyFile = False
        return new_filename, df_dict, cleanMethod_dict, verifyFile

    try:

        if modelType == 'Regression':
            if y.dtypes.name not in numerical_type:
                if y.nunique() < 10:
                    verifyFile = False
                    return new_filename, df_dict, cleanMethod_dict, verifyFile
                else:
                    dataNo = len(y)
                    i = 0
                    ind = 0
                    for val in y:
                        if val == None:
                            a = 1
                            
                        elif str(val).isnumeric() or isfloat(val):
                            i += 1
                            
                        else:
                            y_drop_row.append(ind)

                        ind += 1

                    if i/dataNo < 0.5:
                        verifyFile = False
                        
                        return new_filename, df_dict, cleanMethod_dict, verifyFile

            else:
                if y.nunique() < 5:
                    verifyFile = False
                    return new_filename, df_dict, cleanMethod_dict, verifyFile

            print('regression')

        elif modelType == 'Classification':
            if y.dtypes.name not in categorical_type:
                if y.nunique() > 10:
                    verifyFile = False
                    return new_filename, df_dict, cleanMethod_dict, verifyFile
                        
            else:
                if y.nunique() > 10:
                    verifyFile = False
                    return new_filename, df_dict, cleanMethod_dict, verifyFile

            print('Classification')

        nullList = y.index[y.isnull() == True].tolist()
        for n in nullList:
            y_drop_row.append(n)

        df = df.drop(y_drop_row, axis = 0)
        df = df.reset_index(drop = True)

        # Proceed to data classify and data cleaning

        if dict['index'] != 0:
            df_index_initial[dict['index']] = df[dict['index']]
            df = df.drop([dict['index']], axis = 1)

        cat_null = []
        num_null_col = []

        df_cat, df_num, df_index, df_date = classifyData(df, cleanMethod_dict, dict)

        if len(df_index_initial) > 0:
            for col in df_index_initial:
                df_index[col] = df_index_initial[col]

        if len(df_cat) > 0:
            df_categorical = clean_Cat(df_cat, cleanMethod_dict, cat_null)
        else:
            df_categorical = df_cat

        df_date = cleanDate(df_date, cleanMethod_dict, cat_null)
        df_categorical = df_categorical.drop(cat_null)
        
        if len(df_num) > 0:
            df_numerical = clean_Num(df_num, cleanMethod_dict, cat_null, num_null_col, col_dict)
        else:
            df_numerical = df_num

        if len(df_index) > 0:
            df_index = df_index.drop(cat_null)

        df_list = [df_index,df_numerical,df_categorical,df_date]
        df_concat = []

        for df in df_list:
            if len(df) > 0:
                df = df.reset_index(drop=True)
                df_concat.append(df)

        df_new = pd.concat(df_concat, axis=1, join='inner')
        
        new_filename = fname + '_cleaned.csv'
        # new_filename = 'testing_cleaned_.csv'
        df_new.to_csv(os.path.join('website/static',new_filename),index=False)

    except:
        new_filename = fname + '_cleaned.csv'
        df_categorical, df_numerical, df_index = classifyCleanData2(df, cleanMethod_dict, dict)
        x.updateProgress(20,'Data Cleaning Error')
        # Save file at line below

    df_dict['index'] = df_index.reset_index(drop=True)
    df_dict['numerical'] = df_numerical.reset_index(drop=True)
    df_dict['categorical'] = df_categorical.reset_index(drop=True)
    df_dict['date'] = df_date.reset_index(drop=True)

    return new_filename, df_dict, cleanMethod_dict, verifyFile

# This function is simplified and will be run only when error of the original function occur
def classifyCleanData2(df, cleanMethod_dict, data_dict):

    df_cat = pd.DataFrame()
    df_num = pd.DataFrame()
    df_index = pd.DataFrame()

    df2 = df.copy()
    nullRow = []

    if data_dict['index'] != 0:
        df_index = df2[data_dict['index']]
        df2 = df2.drop([data_dict['index']], axis = 1)


    for col_name in df2:
        col = df2[col_name]
        col_type = col.dtypes.name

        nullList = col.index[col.isnull() == True].tolist()
        if len(nullList) != 0:
            cleanMethod_dict[col_name]['problem'].append("Null Value")
            cleanMethod_dict[col_name]['flag'] = 1
            cleanMethod_dict[col_name]['method'].append("Null Removal")
            for n in nullList:
                if n not in nullRow:
                    nullRow.append(n)

        if col_type in categorical_type:
            df_cat[col_name] = col

        elif col_type in numerical_type:
            if col.nunique() > 10:
                df_num[col_name] = col
            else:
                df_cat[col_name] = col

    if len(df_cat) != 0:
        df_cat = df_cat.astype(str)
        df_cat = df_cat.drop(nullRow, axis = 0)

    if len(df_num) != 0:
        df_num = df_num.drop(nullRow, axis = 0)

    if len(df_index) != 0:
        df_index = df_index.drop(nullRow, axis = 0)

    return df_cat, df_num, df_index

def classifyData(df, cleanMethod_dict, data_dict):

    df_cat = pd.DataFrame()
    df_num = pd.DataFrame()
    df_index = pd.DataFrame()
    df_date = pd.DataFrame()

    isDate = []

    index_exist = data_dict['index']

    for col_name in df:

        # cleanMethod_dict[col_name] = {
        #     'method' : [],
        #     'flag' : 0,
        #     'problem' : []
        # }

        col = df[col_name]
        col_type = col.dtypes.name

        if col_type in numerical_type:
            if (col.fillna(-9999) % 1  == 0).all() and col.nunique() > (len(col) * 0.8):
                df_index[col_name] = col
            elif col.nunique() > 5:
                df_num[col_name] = col
            else:
                df_cat[col_name] = col
            
        elif col_type in categorical_type:
            
            int_Exist = np.any([isinstance(val, int) for val in col])
            
            no_float = 0
            float_Exist = np.any([isinstance(val, float) for val in col])
            
            for n in col:
                match = re.search("\d+[/-]\d+[/-]\d+", str(n))

                if isinstance(n,float):
                    no_float += 1

                elif match:
                    isDate.append(1)

                elif match != True:
                    isDate.append(0)

            if len(isDate) > 0:
                x = isDate.count(1)
                y = isDate.count(0)
            
                if x > y:
                    df_date[col_name] = col
                    continue
                    
            isDate = []
            
            no_Null = col.isnull().sum()

            if col.nunique() > len(col) * 0.8:
                df_index[col_name] = col

            elif no_Null < no_float:
                df_num[col_name] = col
                
            elif no_float == no_Null:
                df_cat[col_name] = col

            elif int_Exist == True and float_Exist == False:
                if col.nunique() > 10:
                        df_num[col_name] = col

                else:
                        df_cat[col_name] = col

            else:
                df_cat[col_name] = col

    return df_cat, df_num, df_index, df_date

def isfloat(num):
    try:
        float(num)
        return True
    except ValueError:
        return False

# Cleaning Method for Numerical

def clean_Num(df_num, cleanMethod_dict, cat_null, num_null_col, col_dict):

    df_num = df_num.drop(cat_null)
    #df_num = df_num.drop(['CLIENTNUM'],axis = 1)

    for col_name in df_num:
        col_dict['original'].append(col_name)
        if df_num[col_name].isnull().sum().item() > 0:
            num_null_col.append(col_name)

    #if df_num.isnull().sum().item() > 0:
    if len(num_null_col) != 0:
        df_num = fillMeanMedian(df_num, cleanMethod_dict, num_null_col, col_dict)

    # df_num = dataTransform(df_num, cleanMethod_dict)

    return df_num

def fillMeanMedian(df_num, cleanMethod_dict, num_null_col, col_dict):
    df = pd.DataFrame()

    for col_name in df_num.loc[:,num_null_col]:

        cleanMethod_dict[col_name]['problem'].append('Null Value')
        cleanMethod_dict[col_name]['method'].append('Fill with Median')

        # new_colName_mean = col_name + '_mean'
        # new_colName_median = col_name + '_median'

        # mean = df_num[col_name].mean()
        median = df_num[col_name].median()

        # df_num[new_colName_mean] = df_num[col_name].fillna(mean)
        # df_num[new_colName_median] = df_num[col_name].fillna(median)

        df_num[col_name] = df_num[col_name].fillna(median)

        # col_dict['mean'].append(new_colName_mean)
        # col_dict['median'].append(new_colName_median)
        cleanMethod_dict[col_name]['flag'] = 1

        # cleanMethod_dict[new_colName_mean] = {
        #     'method' : 'None',
        #     'flag' : 0,
        #     'problem' : []
        # }

        # cleanMethod_dict[new_colName_median] = {
        #     'method' : 'None',
        #     'flag' : 0,
        #     'problem' : []
        # }

    return df_num

def detectOutlier(df_num):
    return True

# Cleaning Method for Categorical

def clean_Cat(df_cat, cleanMethod_dict, cat_null):

    df_cat = drop_catNull(df_cat, cleanMethod_dict, cat_null)
    df_cat = df_cat.astype(str)

    return df_cat

def drop_catNull(df_cat, cleanMethod_dict, cat_null):

    for col_name in df_cat:
        col = df_cat[col_name]
        nullList = col.index[col.isnull() == True].tolist()
        for idx in nullList:
            if idx not in cat_null:
                cat_null.append(idx)
        
        if len(nullList) > 0:
            cleanMethod_dict[col_name]['problem'].append('Null Value')
            cleanMethod_dict[col_name]['method'].append('Removal')
            cleanMethod_dict[col_name]['flag'] = 1

    return df_cat

# Clean date method

def cleanDate(df_date, cleanMethod_dict, cat_null):

    for col_name in df_date:
        col = df_date[col_name]
        nullList = col.index[col.isnull() == True].tolist()
        for idx in nullList:
            if idx not in cat_null:
                cat_null.append(idx)
        
        if len(nullList) > 0:
            cleanMethod_dict[col_name]['problem'].append('Null Value')
            cleanMethod_dict[col_name]['method'].append('Removal')
            cleanMethod_dict[col_name]['flag'] = 1

    df_date = df_date.drop(cat_null)

    return df_date

# df = pd.read_csv('train.csv')
# cleanMethod_dict = {}
# dict = {
#     'target' : 'Churn',
#     'index' : 'CustomerID'
# }
# for col_name in df:
#     cleanMethod_dict[col_name] = {
#         'method' : [],
#         'flag' : 0,
#         'problem' : []
#     }
# df_categorical, df_numerical, df_index = classifyCleanData2(df, cleanMethod_dict, dict)

# print(df_categorical)
# print(df_numerical)
# print(df_index)