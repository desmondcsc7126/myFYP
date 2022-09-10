import pandas as pd
import numpy as np
import os

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def dataTransform(filename, df_dict, cleanMethod_dict, username):

    df_date_extracted = pd.DataFrame()

    df_num = df_dict['numerical']
    df_cat = df_dict['categorical']
    df_date = df_dict['date']
    df_index = df_dict['index']

    target = df_dict['target']

    if len(df_num) > 0:
        df_num = numTransform(df_num, cleanMethod_dict, target)
    
    if len(df_date) > 0:
        df_date_extracted = dateTransform(df_date)

    df_list = [df_index, df_num, df_cat, df_date, df_date_extracted]
    df_concat = []

    for df in df_list:
        if len(df) > 0:
            df_concat.append(df)

    df_new = pd.concat(df_concat, axis=1, join='inner')
    if username == '':
        new_filename = 'transformed_'+ filename + '.csv'
    else:
        new_filename = 'transformed_'+ username + '.csv'

    df_new.to_csv(os.path.join('website/static',new_filename),index=False)

    print('Data Transform Part')
    print(df_date_extracted)
    print(df_new)

    return df_new, df_dict, df_date_extracted, cleanMethod_dict, new_filename

def numTransform(df_num, cleanMethod_dict, target):

    for col_name in df_num:
        if col_name != target:
            skew_score = df_num[col_name].skew()
            print(col_name,' Skew Score:',skew_score)

            zeroExist = np.count_nonzero(df_num[col_name]==0,axis=0)
            negativeExist = np.count_nonzero(df_num[col_name] < 0,axis=0)

            if skew_score > 0.8:

                cleanMethod_dict[col_name]['problem'].append('Positively Skewed')

                if zeroExist == 0 and negativeExist == 0:
                    df_num[col_name] = np.log10(df_num[col_name])
                    cleanMethod_dict[col_name]['method'].append('Log10 Transformation')
                    cleanMethod_dict[col_name]['flag'] = 1

                elif zeroExist != 0 and negativeExist == 0:
                    df_num[col_name] = np.sqrt(df_num[col_name])
                    cleanMethod_dict[col_name]['method'].append('Squareroot Transformation')
                    cleanMethod_dict[col_name]['flag'] = 1

                else:
                    cleanMethod_dict[col_name]['method'].append('No suitable Transformation due to negative values')

            elif skew_score < -0.8:
                cleanMethod_dict[col_name]['problem'].append('Negatively Skewed')
                df_num[col_name] = np.log10(np.max(df_num[col_name]) - df_num[col_name] + 1)
                cleanMethod_dict[col_name]['method'].append('Reflected Log10 Transformation')
                cleanMethod_dict[col_name]['flag'] = 1
            
    return df_num

def dateTransform(df_date):

    df_date_extracted = pd.DataFrame()

    print(df_date)

    month = ['january','february','march','april','may','june','july','august','september','october','november','december']
    month_short = ['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec']

    for key in df_date:

        tokenized_DateList = [[],[],[]]

        ind = ['month','day','year']
        indicator = 0

        date_dict = {
            0:'day',
            1:'month',
            2:'year'
        }

        for val in df_date[key]:
            
            myList = val.split('/')
            if len(myList) <= 1:
                myList = val.split('-')
            
            if indicator != 6:
                
                i = 0       
                for n in myList:
                    if n.lower() in month or n.lower() in month_short:
                        ind[i] = 'month'
                        indicator += 1
                    elif int(n) <= 12:
                        #print('month')
                        ind[i] = 'month'
                        indicator += 1
                    elif int(n) <= 31:
                        #print('day')
                        ind[i] = 'day'
                        indicator += 2
                    elif int(n) > 31:
                        #print('year')
                        ind[i] = 'year'
                        indicator += 3
                    else:
                        ind[i] = 'invalid'
                        
                    i += 1

                if indicator < 6:
                    indicator = 0
            
            tokenized_DateList[0].append(myList[0])
            tokenized_DateList[1].append(myList[1])
            tokenized_DateList[2].append(myList[2])

        print(tokenized_DateList)
        print(date_dict)
        print(ind)

        for key2 in date_dict:
            date_dict[key2] = ind[key2]

        for key2 in date_dict:
            col_name = str(key) + '_' + str(date_dict[key2])
            # df_date[col_name] = tokenized_DateList[key2]
            df_date_extracted[col_name] = tokenized_DateList[key2]

    return df_date_extracted

def scalingLabelling(df_dict, df_date_extracted1, target, x):

    df_modelling_dict = {
        'numerical': 0,
        'categorical': 0,
        'target' : df_dict['target']
    }

    label_class = []

    mmc = MinMaxScaler()
    lb = LabelEncoder()

    df_for_modelling = pd.DataFrame()
    df_cat = pd.DataFrame()
    df_num = pd.DataFrame()
    df_index = pd.DataFrame()
    df_date_extracted = pd.DataFrame()

    drop_col = []

    try:
        if len(df_date_extracted1) > 0:
            df_date_extracted = df_date_extracted1.copy()
            df_date_extracted = df_date_extracted.astype(str)
            for col in df_date_extracted:
                df_date_extracted[col] = lb.fit_transform(df_date_extracted[col])

        if len(df_dict['numerical']) > 0:
            df_num = df_dict['numerical'].copy()

        if len(df_dict['categorical']) >0:
            # df_cat = df_dict['categorical'].copy()
            # for col in df_cat:
            #     df_cat[col] = lb.fit_transform(df_cat[col])
            #     if col == target:
            #         label_class = lb.classes_
            df_cat, label_class = catLabelling(df_dict['categorical'], lb, target)

        if len(df_date_extracted) > 0:
            df_cat = pd.concat([df_cat, df_date_extracted], axis = 1, join='inner')

        if len(df_dict['index']) > 0:
            df_index = df_dict['index'].copy()

        # drop_col = []

        drop_col = featureSelection(df_num, df_cat, df_index, target)

        print('Drop col: ',drop_col)

        df_list = [df_num, df_cat]
        # df_concat = []

        if len(df_num) > 0:
            for col in df_num:
                if col in drop_col:
                    df_num = df_num.drop([col],axis = 1) 

        if len(df_cat) > 0:
            for col in df_cat:
                if col in drop_col:
                    df_cat = df_cat.drop([col],axis = 1) 

        x.updateProgress(30,'Data Labelling and Selection Done')

    except:
        if len(df_dict['categorical']) >0:
            df_cat, label_class = catLabelling(df_dict['categorical'], lb, target)
        if len(df_dict['numerical']) > 0:
            df_num = df_dict['numerical'].copy()
            
        x.updateProgress(30,'Data Labelling and Selection Error')
        

    df_modelling_dict['numerical'] = df_num
    df_modelling_dict['categorical'] = df_cat

    return df_modelling_dict, drop_col, label_class

def catLabelling(df, lb, target):
    label_class = []
    df_cat = df.copy()
    for col in df_cat:
        df_cat[col] = lb.fit_transform(df_cat[col])
        if col == target:
            label_class = lb.classes_
    return df_cat, label_class

def featureSelection(df_num, df_cat, df_index, target):

    dfList = [df_num, df_cat]
    df_concat = []

    for df in dfList:
        if len(df) > 0:
            df_concat.append(df)

    df_check = pd.concat(df_concat, axis=1, join='inner')

    corr = df_check.corr()

    r = 0

    for val in corr[target]:
        r += abs(val)

    drop_col = []
    corrIndex = corr[target].index

    print('Correlation: ',r)

    for key,val in zip(corrIndex,corr[target]):
        print(key,': ',val)
        if abs(val) < r * 0.01:
            drop_col.append(key)

    for col_name in df_check:
        if df_check[col_name].nunique() <= 1:
            if col_name not in drop_col:
                drop_col.append(col_name)

    for col_name in df_index:
        drop_col.append(col_name)

    return drop_col