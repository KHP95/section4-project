from flask import Flask
from flask import render_template
from flask import request
from flask import redirect

import os
import joblib
from datetime import date, datetime

import psycopg2
import pandas as pd
import numpy as np

from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBRegressor

# api는 간단하게 할 예정이므로 블루프린트 사용x

app = Flask(__name__)

FILE_PATH = os.path.dirname(__file__)
conn = None
cursor = None

oe = joblib.load(os.path.join(FILE_PATH, 'encoder.pkl'))
model = joblib.load(os.path.join(FILE_PATH, 'model.pkl'))

def predict(X):
    """
    모델을 통해 예측하는 함수

    args:
        X : features (pandas df)
    
    returns:
        pred : 예측값 (list)
    """
    global oe
    global mode
    
    X = X.copy()
    X.showTm = X.showTm.astype('int')
    X.rating = X.rating.astype('float')

    # openDt는 epoch time으로 변환 (결측치는 nan으로)
    X.openDt = X.openDt.apply(lambda x: datetime.strptime(x, "%Y-%m-%d").date())
    X.openDt = X.openDt.apply(
        lambda x: (x-date(1970,1,1)).total_seconds() if isinstance(x, date) else float('nan')
        )
    cat_cols = ['nations', 'audit', 'director', 'genre1', 'actor1', 'actor2', 'actor3']

    # 'NaN'을 ''으로 변환후 인코딩
    X[cat_cols] = X[cat_cols].apply(lambda x: x.replace('NaN', ''))
    X[cat_cols] = oe.transform(X[cat_cols])

    pred = list(model.predict(X))

    return pred


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/connect', methods=['GET', 'POST'])
def connect():
    global conn
    global cursor

    if request.method == 'POST':
        if 'close_conn' in request.form:
            try:
                conn.close()
                message = "db연결 종료"
                return render_template('connect.html', message=message)
            
            except Exception as e:
                message = f"연결 종료 실패 : {e}"
                return render_template('connect.html', message=message)
            
        try:
            password = request.form['password']
            conn = psycopg2.connect(host="localhost",
                                    dbname='movie_db',
                                    user='postgres',
                                    password=password,
                                    port=5432)
            cursor = conn.cursor()
            message = '연결성공\n' + str(conn)
        except Exception as e:
            message = f'연결실패\n원인 : {e}'
        finally:
            return render_template('connect.html', message=message)
    
    if request.method == 'GET':
        return render_template('connect.html')

@app.route('/sql', methods=['GET', 'POST'])
def sql():
    global conn
    global cursor
    try:
        cond = conn.closed # 0이면 연결, 1이면 연결종료상태
    except:
        cond = None

    if request.method =='GET':
        return render_template('sql.html', cond=cond)
    
    if request.method == 'POST':
        try:
            quote = request.form['sql_quote']
            # SELECT 구문이면 result 표시
            if 'SELECT' in quote:
                cursor.execute(quote)
                result = pd.DataFrame(
                    cursor.fetchall(),
                    columns=[i[0] for i in cursor.description]
                    ).to_html(justify='center')
                message = '성공'
                
                return render_template('sql.html', cond=cond, message=message,
                               result=result)
            
            # 그외 구문(DDL, DML)이면 result 없이 성공여부만 출력
            cursor.execute("BEGIN")
            cursor.execute(quote)
            cursor.execute("COMMIT")
            result = ''
            message = '성공'

        except Exception as e:
            conn.rollback() if conn else None
            result = ''
            message = f'실패 : {e}'
        
        return render_template('sql.html', cond=cond, message=message,
                               result=result)

@app.route('/predict', methods=['POST', 'GET'])
def model_page():
    if request.method == 'GET':
        return render_template('predict.html')
    
    if request.method == 'POST':
        col_name = ['showTm', 'openDt', 'rating', 'nations',\
                    'audit', 'director', 'genre1', 'actor1',\
                    'actor2', 'actor3']
        try:
            temp = [[request.form.get(i) for i in col_name]]
            X = pd.DataFrame(temp, columns=col_name)
            print(X.iloc[0].to_list())
            pred = predict(X)
            pred = int(pred[0])
            message = f"작업성공. 예측값 : {pred}명"
            
            return render_template('predict.html', message=message)
        
        except Exception as e:
            message = f"작업실패 : {e}"
            return render_template('predict.html', message=message)
        
@app.route('/dashboard', methods=['GET'])
def dashboard():
    # metabase로 redirect
    return redirect('http://127.0.0.1:3000/dashboard/1')
    

        