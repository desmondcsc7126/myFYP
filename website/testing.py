# from dataCleaning import cleanData
# from dataTransform import dataTransform, scalingLabelling
# from dataModelling import dataModelling
# from regression import regression
import pandas as pd
import random
import json
import threading
import time

class Progress:
    def __init__(self):
        self.progress = 0
        self.status = []

    def updateProgress(self, progress, status):
        self.progress = progress
        self.status.append(status)
        return True

    def getProgress(self):
        return self.progress

    def getStatus(self):
        return self.status

x = Progress()

# def main():

#     df = pd.read_csv('insurance.csv')
#     filename = 'myProject'

#     modelType = 'Regression'
#     # modelChoice = ['KNN', 'LGR', 'RFC', 'NB', 'DT', 'XGBoost', 'SVM']
#     modelChoice = ['DTR']
#     # modelChoice = ['LR','DTR','KNNR','PLR']

#     # df['Balance']=df['Balance'].replace(0,df['Balance'].median())

#     dict = {
#         'index' : 0,
#         'target' : 'charges'
#     }

#     fname, df_dict, cleanMethodDict, verifyFile = cleanData(df, filename, dict, modelType, x)
#     x.updateProgress(20,'Data Cleaning Done')

#     df, df_dict, df_date_extracted, cleanMethodDict = dataTransform(fname, df_dict, cleanMethodDict)
#     x.updateProgress(30,'Data Transformation Done')

#     df_modelling_dict, drop_col, label_class = scalingLabelling(df_dict, df_date_extracted, dict['target'], x)
#     x.updateProgress(30,'Data Modelling in progress.....')

#     print('Index: ',df_dict['index'])
#     print('Num: ',df_dict['numerical'])
#     print('Cat: ',df_dict['categorical'])
#     print('Date: ',df_dict['date'])

#     # print('Whole File After Transformation: ',df)
#     # print(cleanMethodDict)

#     print('File after Scaling:')
#     print(df_modelling_dict)
#     print(df_modelling_dict['numerical'])
#     print(df_modelling_dict['categorical'])

#     if modelType == 'Classification':
#         final_model_dict = dataModelling(df_modelling_dict,modelChoice, x, modelType, label_class)
#     else:
#         final_model_dict = regression(df_modelling_dict,modelChoice, x, modelType)

#     # print(final_model_dict)

#     return True

# main()

def myLongTask(q):
    time.sleep(5)
    print(q)
    return x

def main2():
    thread = threading.Thread(target = myLongTask, args=(5,))
    thread.start()
    print("main2")
    return True

main2()