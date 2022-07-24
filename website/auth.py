import random
# from re import X
from flask import Blueprint, render_template, request
import pandas as pd
import time
import json
import os
import threading
from pymysql import connections
import boto3
import datetime
import re

from .dataCleaning import cleanData
from .dataTransform import dataTransform, scalingLabelling
from .dataModelling import dataModelling
from .regression import regression

auth = Blueprint('auth', __name__)

customuser = "desmondcsc"
custompass = "desmondcsc"
customdb = "users"
customhost = "users.cvdg99ehiopr.us-west-2.rds.amazonaws.com" 
custombucket = "desmondchongsoonchuen"

db_conn = connections.Connection(
    host=customhost,
    port=3306,
    user=customuser,
    password=custompass,
    db=customdb
)

s3 = boto3.resource(
    's3'
)

my_bucket = s3.Bucket(custombucket)

s3_client = boto3.client(
    's3'
)

# Insert object into s3 bucket
# s3.Bucket(custombucket).put_object(Key=emp_photo_name, Body=emp_photo)

# Delete object from s3 bucket
# for file in my_bucket.objects.all():
#     if file.key.startswith(strData):
#         client.delete_object(Bucket = custombucket, Key = file.key)

class Progress:
    def __init__(self):
        self.progress = 0
        self.status = []
        self.taskFinish = False
        self.summaryDict = {}

    def updateProgress(self, progress, status):
        self.progress = progress
        self.status.append(status)
        return True

    def updateTaskFinish(self, taskFinish, summaryDict):
        self.taskFinish = taskFinish
        self.summaryDict = summaryDict
        return True

    def getProgress(self):
        return self.progress

    def getStatus(self):
        return self.status

    def getTaskFinish(self):
        return self.taskFinish

    def getSummaryDict(self):
        return self.summaryDict

progressArr = [None] * 1000
occupiedIndex = []

@auth.route('/beforeModelling', methods = ['POST',"GET"])
def test1():
    username = request.form['username']
    return render_template('beforeModelling.html', username = username)

def mylastHope():
    return render_template('test2.html')

@auth.route('/registration',methods = ['POST', 'GET'])
def register():
    return render_template('registration.html', repeated = "False")

@auth.route('/registerUser', methods=['POST','GET'])
def registerUser():

    unArr = []
    email = request.form['email']
    username = request.form['username']
    password = request.form['password']

    # Should check repeated username
    cursor = db_conn.cursor()
    sql = "SELECT username FROM users"
    cursor.execute(sql)
    data = cursor.fetchall()
    data = list(map(list,data))
    cursor.close()

    for n in data:
        unArr.append(n[0])

    if username in unArr:
        return render_template('registration.html', repeated = "True", data = data)

    # Insert
    insert_user = "INSERT INTO users VALUES (%s, %s, %s)"

    cursor = db_conn.cursor()

    try:
        cursor.execute(insert_user,(username, email, password))
        db_conn.commit()

    finally:
        cursor.close()

    return render_template('login.html', success = "True", data = unArr)

@auth.route('/login', methods = ['POST','GET'])
def login():
    return render_template('login.html')

@auth.route('/authentication',methods = ['POST','GET'])
def authentication():

    username1 = request.form['username']
    password1 = request.form['password']

    cursor = db_conn.cursor()
    sql = "SELECT password from users where username = %s"
    cursor.execute(sql,(username1, ))
    # cursor.execute(sql)
    data = cursor.fetchone()
    # data = list(map(list,data))
    cursor.close()

    if data != None:
        pw = data[0]
        if password1 != pw:
            return render_template('login.html', success = "False", login = "False")
    else:
        return render_template('login.html', success = "False", login = "False")

    return render_template('base.html', username = username1)

@auth.route('/logout', methods = ['POST','GET'])
def logout():
    return render_template('base.html')

@auth.route('/visualisation',methods = ['POST', 'GET'])
def visualisation():
    return render_template('visualisation.html')

@auth.route('/checkFileForVisualise',methods = ['POST','GET'])
def check():
    file = request.form['file']
    df = pd.read_csv(file)

    myDict = {
        'verify' : True,

        'numerical' : {

        },

        'categorical' : {

        }      
    }

    numerical_type=['int64','float64']
    categorical_type=['object']

    filename = file.filename
    name, extension = os.path.splitext(filename)

    if extension != ".csv":
        myDict['verify'] = False
        myDict['Error Message'] = "File is not in .csv format. Please upload again."
        return myDict

    for col in df:
        if df[col].dtypes.name in numerical_type:

            colDict = {
                'Null' : df[col].isnull().sum(),
                'Skewness' : df[col].skew(),
                'Value' : []
            }

            for val in df[col]:
                colDict['Value'].append(val)

            myDict['numerical'][col] = colDict

        else:
            colDict = {
                'Null' : df[col].isnull().sum(),
                'Value' : []
            }

            for val in df[col]:
                myDict['categorical'][col] = colDict
            
    return myDict

@auth.route('/createProgress', methods = ['POST','GET'])
def createProgress():
    validIndex = "False"

    while validIndex == "False":
        myIndex = random.randint(0,999)
        if myIndex in occupiedIndex:
            validIndex = "False"
        else:
            x = Progress()
            progressArr[myIndex] = x
            occupiedIndex.append(myIndex)
            validIndex = "True"

    return str(myIndex)

@auth.route('/returnProgress', methods = ['POST','GET'])
def returnProgress():
    myIndex = int(request.form['index'])
    x = progressArr[myIndex]

    myDict = {
        'status' : [],
        'taskFinish' : "False",
        'summaryDict' : {}
    }

    if x.getTaskFinish() == False:
        myDict['status'] = x.getStatus()
        return myDict
    else:
        myDict['status'] = x.getStatus()
        myDict['taskFinish'] = "True"
        myDict['summaryDict'] = x.getSummaryDict()
        return myDict

    # status = x.getStatus()
    # return json.dumps(status)

@auth.route('/deleteProgressClass', methods = ['POST','GET'])
def deleteprogress():
    myIndex = int(request.form['index'])
    # progressArr.pop(myIndex)
    progressArr[myIndex] = None
    occupiedIndex.remove(myIndex)
    return str(occupiedIndex)

@auth.route('/verifyFile',methods = ['POST','GET'])
def verifyFile():

    rspd = {
        'verify' : True,
        'col' : []
    }
    f = request.files['file']
    filename = f.filename
    name, extension = os.path.splitext(filename)

    if extension != ".csv":
        rspd['verify'] = False
        rspd['Error Message'] = "File is not in .csv format. Please upload again."
        return rspd

    else:
        df = pd.read_csv(f)
        colList = []
        for col in df:
            colList.append(col)

        rspd['col'] = colList
        return rspd

# @auth.route('/cleanData', methods = ['POST','GET'])
# def cleanDataMain():

#     f = request.files['file']
#     dict = json.loads(request.form['dict'])
#     myIndex = int(request.form['index'])
#     model_choice = json.loads(request.form['modelChoice'])
#     modelType = request.form['modelType']
#     filename = f.filename

#     df = pd.read_csv(f)

#     x = progressArr[myIndex]

#     summary_dict = {

#     }

#     fname, df_dict, cleanMethodDict, verifyFile = cleanData(df, filename, dict, modelType, x)

#     if verifyFile == False:
#         summary_dict['verify'] = False
#         return summary_dict
        
#     x.updateProgress(20,'Data Cleaning Done')

#     try:
#         df, df_dict, df_date_extracted, cleanMethodDict = dataTransform(fname, df_dict, cleanMethodDict)
#         x.updateProgress(30,'Data Transformation Done')
#     except:
#         print("Error")
#         df_date_extracted = pd.DataFrame()
#         x.updateProgress(30,'Data Transformation Error')

#     df_modelling_dict, drop_col, label_class = scalingLabelling(df_dict, df_date_extracted, dict['target'], x)
#     x.updateProgress(30,'Data Modelling in progress.....')

#     if modelType == 'Classification':
#         final_model_dict = dataModelling(df_modelling_dict,model_choice, x, modelType, label_class)
#     else:
#         final_model_dict = regression(df_modelling_dict,model_choice, x, modelType)

#     summary_dict = {
#         'model_dict': final_model_dict.copy(),
#         'data_dict' : cleanMethodDict.copy(),
#         'drop_col' : drop_col.copy(),
#         'verify' : True
#     }

#     return summary_dict

@auth.route('/cleanData', methods = ['POST','GET'])
def cleanDataMain():

    f = request.files['file']
    dict = json.loads(request.form['dict'])
    myIndex = int(request.form['index'])
    model_choice = json.loads(request.form['modelChoice'])
    modelType = request.form['modelType']
    username = request.form['username']
    filename = f.filename

    df = pd.read_csv(f)

    x = progressArr[myIndex]

    fname, df_dict, cleanMethodDict, verifyFile = cleanData(df, filename, dict, modelType, x)

    if verifyFile == False:
        return "False"
        
    x.updateProgress(20,'Data Cleaning Done')

    thread = threading.Thread(target=backgroundTask, args=(x,dict,model_choice,modelType,fname,df_dict,cleanMethodDict,username,))
    thread.start()

    return "True"

def backgroundTask(x, dict, model_choice, modelType, fname, df_dict, cleanMethodDict, username):

    summary_dict = {

    }

    try:
        df, df_dict, df_date_extracted, cleanMethodDict = dataTransform(fname, df_dict, cleanMethodDict)
        x.updateProgress(30,'Data Transformation Done')
    except:
        print("Error")
        df_date_extracted = pd.DataFrame()
        x.updateProgress(30,'Data Transformation Error')

    df_modelling_dict, drop_col, label_class = scalingLabelling(df_dict, df_date_extracted, dict['target'], x)
    x.updateProgress(30,'Data Modelling in progress.....')

    if modelType == 'Classification':
        final_model_dict = dataModelling(df_modelling_dict,model_choice, x, modelType, label_class)
    else:
        final_model_dict = regression(df_modelling_dict, model_choice, x, modelType)

    summary_dict = {
        'model_dict': final_model_dict.copy(),
        'data_dict' : cleanMethodDict.copy(),
        'drop_col' : drop_col.copy(),
        'verify' : True
    }

    x.updateTaskFinish(True,summary_dict)

    if username != '':
        # Store file in s3
        cursor = db_conn.cursor()
        sql = "select count(*) from modelHist where username = %s"
        cursor.execute(sql,(username, ))
        data = cursor.fetchone()
        seq = data[0] + 1
        cursor.close()

        dictName = username + "_" + str(seq) + '.json'
        s3.Bucket(custombucket).put_object(Key=dictName, Body=json.dumps(summary_dict))

        # Store filename in database
        insert_sql = "INSERT INTO modelHist VALUES (%s, %s, %s, %s, %s)"
        
        now = str(datetime.datetime.now())

        cursor = db_conn.cursor()

        try:
            cursor.execute(insert_sql,(username, now, fname, dictName, modelType))
            db_conn.commit()

        finally:
            cursor.close()

    return True

@auth.route('/modelHist', methods = ['POST','GET'])
def modelHist():

    username = request.form['username']
    hist_dict = {}

    # for file in my_bucket.objects.all():
    #     filename = file.key
    #     if re.search("{un}.+{fileformat}$".format(un = username, fileformat = ".json"),filename):
    #         fileList.apend(filename)

    sql = "select * from modelHist where username = %s"
    cursor = db_conn.cursor()
    cursor.execute(sql,(username, ))
    data = cursor.fetchall()
    
    i = 1
    for n in data:

        temp_dict = {
            'time' : n[1],
            'filename' : n[2],
            'dictID' : n[3],
            'modelType' : n[4]
        }
        hist_dict[i] = temp_dict.copy()
        i += 1

    return render_template('modelHist.html', hist_dict = hist_dict, username = username)

@auth.route('/generalReport',methods = ['POST', 'GET'])
def generalReport():

    myResult = json.loads(request.form['myResult'])
    username = request.form['username']
    myResult = myResult['summaryDict']
    # myResult = request.form['myResult']

    # myResult = {
    #     'KNN' : {
    #         'Parameter List' : {},
    #         'Optimal Parameter' : {'weights': 'uniform', 'n_neighbors': 37, 'p': 1},
    #         'Accuracy': round(0.786923076923077,4), 
    #         'Recall': round(0.5384615384615384,4), 
    #         'Precision': round(0.6426229508196721,4)
    #     }
    # }
    modelType = myResult['model_dict']['modelType']
    modelResult = myResult['model_dict']['model']
    dataClean = myResult['data_dict']
    drop_col = myResult['drop_col']

    return render_template('generalReport.html',myResult = modelResult, dataClean = dataClean, drop_col = drop_col, modelType = modelType, username = username)

@auth.route('/generalReport2', methods = ['POST','GET'])
def generalReport2():
    dictID = str(request.form['dictID'])
    username = request.form['username']
    response = s3_client.get_object(Bucket = custombucket, Key = dictID)
    myResult = json.loads(response['Body'].read())

    modelType = myResult['model_dict']['modelType']
    modelResult = myResult['model_dict']['model']
    dataClean = myResult['data_dict']
    drop_col = myResult['drop_col']

    return render_template('generalReport.html',myResult = modelResult, dataClean = dataClean, drop_col = drop_col, modelType = modelType, username = username)

@auth.route('/detailReport',methods = ['POST','GET'])
def detailReport():

    mydict = str(request.form['modelDict'])
    mydict = mydict.replace("'",'"')
    mydict = json.loads(mydict)

    modelName = str(request.form['mName'])
    username = request.form['username']

    return render_template('detailReport.html', mydict = mydict, modelName = modelName, username = username)

# Junk ......

@auth.route('/test2', methods = ['POST'])
def test2():
    input = request.form['test']
    return render_template('test2.html', input = input)

@auth.route('/progress', methods = ['POST','GET'])
def progress():

    myTimer = 1
    # x = Progress()

    while myTimer <= 10:
        time.sleep(1)
        x.updateProgress(myTimer, None)
        myTimer += 1

    return "Success"

@auth.route('/test3',methods = ['POST'])
def test3():
    return "bye"

@auth.route('/uploadImg',methods = ['POST',"GET"])
def uploadImg():
    if request.method=='POST':
        f=request.files['img']
    return render_template('test1.html',filename=f.filename)