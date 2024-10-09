from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import pickle
import sklearn
print(sklearn.__version__)
#loading models
dtr = pickle.load(open('dtr.pkl','rb'))
preprocessor = pickle.load(open('preprocessor.pkl','rb'))

#flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')
@app.route("/predict",methods=['POST'])
def predict():
    if request.method == 'POST':
        Year = request.form['Year']
        average_rain_fall_mm_per_year = request.form['average_rain_fall_mm_per_year']
        pesticides_tonnes = request.form['pesticides_tonnes']
        avg_temp = request.form['avg_temp']
        Area = request.form['Area']
        Item  = request.form['Item']

            # Prepare the features as a DataFrame (ensure the column names match what the preprocessor expects)
        features = pd.DataFrame({
            'Year': [Year],
            'average_rain_fall_mm_per_year': [average_rain_fall_mm_per_year],
            'pesticides_tonnes': [pesticides_tonnes],
            'avg_temp': [avg_temp],
            'Area': [Area],
            'Item': [Item]
        })
        transformed_features = preprocessor.transform(features)
        prediction = dtr.predict(transformed_features).reshape(1,-1)

        return render_template('index.html', prediction = prediction)

if __name__=="__main__":
    app.run(debug=True)