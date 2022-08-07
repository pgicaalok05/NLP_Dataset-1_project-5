import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template

import pickle

app = Flask(__name__)
model = pickle.load(open('project_nlp-5.pkl','rb'))  

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
print(cv)
corpus=pd.read_csv('corpus_dataset-project-5.csv')

corpus1=corpus['corpus'].tolist()
X = cv.fit_transform(corpus1).toarray()


@app.route('/')
def home():
  
    return render_template("index.html")
  
@app.route('/predict',methods=['GET'])
def predict():
        
    '''
    For rendering results on HTML GUI
    '''
    text = request.args.get('text')
    text=[text]
    input_data = cv.transform(text).toarray()
    
    prediction = model.predict(input_data)
    if prediction==1:
      result="Positive"
    else:
      result="Negative"
            
    return render_template('index.html', prediction_text='NLP Model  has predicted about the text : {}'.format(result))

if __name__=="__main__":
  app.run(debug=True)




