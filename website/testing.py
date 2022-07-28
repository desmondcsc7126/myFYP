# from dataCleaning import cleanData
# from dataTransform import dataTransform, scalingLabelling
# from dataModelling import dataModelling
# from regression import regression
import pandas as pd
import random
import json
import threading
import time
from pymysql import connections
import boto3
import re
import smtplib
from itsdangerous import URLSafeTimedSerializer

s = URLSafeTimedSerializer('MySecretCode')

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

# cursor = db_conn.cursor()
# sql = "select count(*) from users"
# cursor.execute(sql)
# data = cursor.fetchone()
# print(data[0] + 1)
# cursor.close()

# print("desmondcsc_" + "1")
# arr = []

# for file in my_bucket.objects.all():
#     arr.append(file)

# print(arr)

# response = s3_client.get_object(Bucket = custombucket, Key = 'desmondcsc_1.json')
# print(json.loads(response['Body'].read()))

# username = "desmondcsc"
# filename = "desmondcsc_1.json"
# x = re.search("{un}.+{fileformat}$".format(un = username, fileformat = ".json"), filename)
# if x:
#     print("True")
# else:
#     print("False")

# hist_dict = {

# }

# sql = "select * from modelHist where username = %s"
# username = "desmondcsc"
# cursor = db_conn.cursor()
# cursor.execute(sql,(username, ))
# data = cursor.fetchall()
# print(data)

# i = 1
# for n in data:

#     temp_dict = {
#         'username' : n[0],
#         'time' : n[1],
#         'filename' : n[2],
#         'dictID' : n[3],
#         'modelType' : n[4]
#     }
#     hist_dict[i] = temp_dict.copy()
#     i += 1

# print(hist_dict)

# gmail_user = 'mytestingemailfyp@gmail.com'
# gmail_password = 'fcuubkfvuehrmipc'

# sent_from = gmail_user
# to = ['bayotir154@galotv.com']
# subject = 'Lorem ipsum dolor sit amet'
# body = 'consectetur adipiscing elit'

# email_text = """\
# From: %s
# To: %s
# Subject: %s

# %s
# """ % (sent_from, ", ".join(to), subject, body)

# try:
#     smtp_server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
#     smtp_server.ehlo()
#     smtp_server.login(gmail_user, gmail_password)
#     smtp_server.sendmail(sent_from, to, email_text)
#     smtp_server.close()
#     print ("Email sent successfully!")
# except Exception as ex:
#     print ("Something went wrong….",ex)

# username = 'desmondcsc'
# cursor = db_conn.cursor()
# sql = "select email from users where username = %s"
# cursor.execute(sql,(username, ))
# email = cursor.fetchone()[0]
# print(email)

# token = s.dumps(email, salt = 'confirmation_email')
# print(token)

link = "https://desmond.com"
# message = "Please click on {link} to verify.".format(link = link)
# print(message)


html = '''
    <html>
        <body>
            <h1>Daily S&P 500 prices report</h1>
            <p>Hello, welcome to your report!</p>
            <a href="{}"></a>
        </body>
    </html>
    '''.format(link)

print(html)
