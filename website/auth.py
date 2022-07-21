from flask import Blueprint, render_template, request, url_for, redirect
from pymysql import connections
import os
import boto3

auth = Blueprint('auth', __name__)

customuser = "desmondcsc"
custompass = "desmondcsc"
customdb = "users"
customhost = "users.cvdg99ehiopr.us-west-2.rds.amazonaws.com" 

db_conn = connections.Connection(
    host=customhost,
    port=3306,
    user=customuser,
    password=custompass,
    db=customdb
)

@auth.route('/registration',methods = ['POST', 'GET'])
def register():
    return render_template('registration.html', repeated = "False")

@auth.route('/registerUser',methods = ['POST','GET'])
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

    return render_template('login.html', success = "False", login = "True")