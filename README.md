#!/bin/bash
sudo yum update -y
sudo yum install git -y
git clone https://github.com/desmondcsc7126/myFYP.git
cd myFYP
sudo pip3 install flask
sudo pip3 install pymysql
sudo pip3 install boto3
sudo pip3 install numpy
sudo pip3 install pandas
sudo pip3 install scikit-learn
sudo pip3 install xgboost
sudo pip3 install itsdangerous
sudo python3 application.py
