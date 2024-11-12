import pickle
from flask import jsonify
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

def getResult(data,model,technique):
    model = model
    # load the model from disk
    filename = 'models/'+str(technique)+'/model_no_seg_'+str(model)+'.sav'
    clf = pickle.load(open(filename, 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))
    X_new = scaler.transform(data)
    print(X_new)
    try:
        #print('try')
        prediction = clf.predict(data)
        #pred =
        #print(clf.classes_)
        probabilities = clf.predict_proba(data)
        #print(probabilities[0][0])
        if model=='knn' or model=='mlp':
            return probabilities
        elif model=='svm':
            return prediction
        else:
            importance = clf.feature_importances_
            return probabilities,importance
        '''if prediction == 0:
            return "Normal: " + str(round((probabilities[0][0]*100),2))+"% , Myocardial Infarction: "+str(round((probabilities[0][1]*100),2))+"% and Myocarditis: "+ str(round((probabilities[0][2]*100),2))+"%"
        elif prediction == 1:
            #return "Myocardium"
            return "Normal: " + str(round((probabilities[0][0]*100),2))+"% , Myocardial Infarction: "+str(round((probabilities[0][1]*100),2))+"% and Myocarditis: "+ str(round((probabilities[0][2]*100),2))+"%"
        elif prediction == 2:
            #print(probabilities[0][2]*100)
            #return "Myocarditis: " + str(probabilities[0][2]*100)+"%"
            return "Normal: " + str(round((probabilities[0][0]*100),2))+"% , Myocardial Infarction: "+str(round((probabilities[0][1]*100),2))+"% and Myocarditis: "+ str(round((probabilities[0][2]*100),2))+"%"
        elif prediction == 0 and model == 'svm':
            return "Normal"
        elif prediction == 1 and model == 'svm':
            return "Myocardial Infarction"
        elif prediction == 2 and model == 'svm':
            return "Myocarditis"'''
    except:
        return "Unable to Detect"
