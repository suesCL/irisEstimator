from flask import Flask, render_template, request
from wtforms import Form, StringField, validators
import pickle
import sqlite3
import os
import numpy as np


app = Flask(__name__)


# Prepare the classifier 
cur_dir = os.path.dirname(__file__)
clf = pickle.load(open(os.path.join(cur_dir,'pkl_objects/classifier.pkl'), 'rb'))


def classify(sizeList):
    label = {0: 'I. setosa', 1: 'I. versicolor', 2: 'I. virginica'}
    X = [sizeList]
    y = clf.predict(X)[0]
    proba =  clf.predict_proba(X).max()
    return label[y], proba

# def train(document, y):
    # X = vect.transform([document])
    # clf.partial_fit(X, [y])





#############################################


class irisForm(Form):
    petal_length = StringField(u'petal_length', [validators.DataRequired()])
    petal_width = StringField(u'petal_width', [validators.DataRequired()])
    sepal_length = StringField(u'sepal_length', [validators.DataRequired()])
    sepal_width = StringField(u'sepal_width', [validators.DataRequired()])
    
    
    
        
@app.route('/')
def index():
    form = irisForm(request.form)
    return render_template('app.html', form = form)


@app.route('/result', methods=['POST'])
def result():
    form = irisForm(request.form)
    if request.method == 'POST' and form.validate():
        petal_length = float(request.form['petal_length'])
        petal_width = float(request.form['petal_width'])
        sepal_length = float(request.form['sepal_length'])
        sepal_width = float(request.form['sepal_width'])
        irisSize = [petal_length, petal_width, sepal_length, sepal_width]

        y,proba = classify(irisSize)

        return render_template('result.html', content = irisSize, prediction = y, probability = round(proba*100,2))

    return render_template('app.html', form=form)

    
    
    
    
    
if __name__ == '__main__':
    app.run(debug = True)
    
