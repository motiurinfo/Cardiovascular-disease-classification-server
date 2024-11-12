import os
import pathological_detection
from flask import Flask
from flask import (
    Blueprint, flash, g, redirect, render_template, request, session, url_for
)
import numpy as np
import pandas as pd
from flask import jsonify
from werkzeug.utils import secure_filename
from sklearn.preprocessing import StandardScaler
from flaskext.mysql import MySQL
import json
app = Flask(__name__)
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'PassWord'
app.config['MYSQL_DB'] = 'pathological_detection'
#Feature Importance
mysql = MySQL(app)

UPLOAD_FOLDER= '/files'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif','py'])
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/result')
def result():
    data = []
    finalResult = {}
    #sex	age	tobacco	overweight	hypertension
    #diabetes	familial_disease	troponin	FEVG	NTP
    model = request.args['model']
    technique = request.args['technique']
    data.append(request.args['sex'])
    data.append(request.args['age'])
    data.append(request.args['tobacco'])
    data.append(request.args['overweight'])
    data.append(request.args['hypertension'])
    data.append(request.args['diabetes'])
    data.append(request.args['FamilialDisease'])
    data.append(request.args['troponin'])
    data.append(request.args['fevg'])
    data.append(request.args['ntp'])
    data = np.array(data).reshape(1,-1)
    if model=='knn' or model=='mlp':
        result  = pathological_detection.getResult(data,model,technique)
        result_string = "Normal: " + str(round((result[0][0]*100),2))+"% , Myocardial Infarction: "+str(round((result[0][1]*100),2))+"% and Myocarditis: "+ str(round((result[0][2]*100),2))+"%"
    elif model == 'svm':
        prediction = pathological_detection.getResult(data,model,technique)
    else:
        result,importance  = pathological_detection.getResult(data,model,technique)
        features = ['sex','age','tobacco','overweight','hypertension','diabetes','familial_disease' ,'troponin','FEVG','NTP']
        importance = [float(i) for i in importance]
        mapping_features = zip(features,importance)
        #print(mapping_features)
        result_string = "Normal: " + str(round((result[0][0]*100),2))+"% , Myocardial Infarction: "+str(round((result[0][1]*100),2))+"% and Myocarditis: "+ str(round((result[0][2]*100),2))+"%"
        features_dictionary = dict(mapping_features)
        print(features_dictionary)
        finalResult['result'] = result_string
        finalResult['features'] = features_dictionary


    #print(finalResult)

    if model=='knn' or model=='mlp':
        return result_string
    elif model =='svm':
        if prediction == 0:
            return "Normal"
        if prediction == 1:
            return "Myocardial Infarction"
        if prediction == 2:
            return "Myocarditis"
    else:
        return finalResult

# @app.route('/upload')
# def upload():
# 	return 'yes'

@app.route('/detection', methods = ['GET', 'POST'])
def detection():
	if request.method == 'POST':
		if 'file' not in request.files:
			flash('no file part')
			return "false"
		file = request.files['file']
		if file.filename == '':
			flash('no select file')
			return 'false'
		if file and allowed_file(file.filename):
			filename = secure_filename(file.filename)
			contents = file.read()
			with open("files/datas.txt","wb") as f:
				f.write(contents)
			file.save = (os.path.join(app.config['UPLOAD_FOLDER'], filename))
			return render_template("getInput.html")
	return  render_template("getInput.html")


@app.route('/add_patients', methods = ['GET', 'POST'])
def patients_info():
    #if request.method == 'POST':

    return  render_template("add_patients.html")

@app.route('/')
def hello():
    return  render_template("first_page.html")


if __name__ == '__main__':
    app.run(debug=True)
