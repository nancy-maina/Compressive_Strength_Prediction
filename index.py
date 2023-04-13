import numpy as np
import pandas as pd
import tensorflow as tf

from flask import Flask, render_template, request, session, abort
from sklearn.preprocessing import normalize

app = Flask(__name__)

model = tf.keras.models.load_model('my_model.h5', compile= False)
model.load_weights('my_model_weights.h5')
model.compile()

@app.route('/')
def index():
    return render_template('index.html', errors={}, form={})

@app.route('/predict', methods=['POST'])
def predict():
    # Validate inputs from the form
    # cement, blast_furnace_slag, fly_ash, water, superplasticizer, coarse_aggregate, fine_aggregate, age
    valid_inputs = {
        'cement': (0, 540),
        'blast_furnace_slag': (0, 359),
        'fly_ash': (0, 200),
        'water': (0, 247),
        'superplasticizer': (0, 32),
        'coarse_aggregate': (0, 1145),
        'fine_aggregate': (0, 992),
        'age': (0, 365)
    }

    errors = {}
    for key, value in valid_inputs.items():
        if not request.form.get(key) or not value[0] <= float(request.form.get(key)) <= value[1]:
            errors[key] = f"Please enter a valid value for {key}. Valid values are between {value[0]} and {value[1]}."

    if errors:
        return render_template('index.html', errors=errors, form=request.form)

    # [cement, blast_furnace_slag, fly_ash, water, superplasticizer, coarse_aggregate, fine_aggregate, age]

    cement = float(request.form.get('cement'))
    blast_furnace_slag = float(request.form.get('blast_furnace_slag'))
    fly_ash = float(request.form.get('fly_ash'))
    water = float(request.form.get('water'))
    superplasticizer = float(request.form.get('superplasticizer'))
    coarse_aggregate = float(request.form.get('coarse_aggregate'))
    fine_aggregate = float(request.form.get('fine_aggregate'))
    age = float(request.form.get('age'))

    # Normalize the inputs
    f_list = [cement, blast_furnace_slag, fly_ash, water, superplasticizer, coarse_aggregate, fine_aggregate, age]
    final_features = normalize([f_list])
    concrete_data = pd.DataFrame(final_features)

    # Predict the compressive strength of concrete using the trained model
    predicted = model.predict(concrete_data)
    predicted = [float(np.round(x)) for x in predicted][0]

    # Calculate the result using the formula
    # Strength = 2.12 * (cement + blast_furnace_slag + fly_ash + water + superplasticizer + coarse_aggregate + fine_aggregate) + 0.001 * age
    calculated =  2.12 * (cement + blast_furnace_slag + fly_ash + water + superplasticizer + coarse_aggregate + fine_aggregate) + 0.001 * age

    # Calculate the difference between the two results, i.e. the error, accuracy, mean absolute error, etc.
    difference = predicted - calculated
    accuracy = (predicted / calculated) * 100
    error = difference / calculated
    mae = abs(difference)
    mse = difference ** 2

    return render_template('prediction.html', predicted=predicted, calculated=calculated, difference=difference, accuracy=accuracy, mae=mae, mse=mse, error=error)

@app.route('/chart')
def chart():
    return render_template('chart.html')


# Chart data
@app.route('/chart_data', methods=['GET'])
def chart_data():
    # Validate inputs from the form
    # cement, blast_furnace_slag, fly_ash, water, superplasticizer, coarse_aggregate, fine_aggregate
    valid_inputs = {
        'cement': (0, 540),
        'blast_furnace_slag': (0, 359),
        'fly_ash': (0, 200),
        'water': (0, 247),
        'superplasticizer': (0, 32),
        'coarse_aggregate': (0, 1145),
        'fine_aggregate': (0, 992),
    }

    errors = {}
    for key, value in valid_inputs.items():
        if not request.args.get(key) or not value[0] <= float(request.args.get(key)) <= value[1]:
            errors[key] = f"Please enter a valid value for {key}. Valid values are between {value[0]} and {value[1]}."

    if errors:
        # Set the status code to 400 and return the errors
        return {'errors': errors}, 400

    # Generate some data given the inputs
    cement = float(request.args.get('cement'))
    blast_furnace_slag = float(request.args.get('blast_furnace_slag'))
    fly_ash = float(request.args.get('fly_ash'))
    water = float(request.args.get('water'))
    superplasticizer = float(request.args.get('superplasticizer'))
    coarse_aggregate = float(request.args.get('coarse_aggregate'))
    fine_aggregate = float(request.args.get('fine_aggregate'))
    
    # Get results for the next 30 days
    results = []

    for i in range(30):
        # Normalize the inputs
        f_list = [cement, blast_furnace_slag, fly_ash, water, superplasticizer, coarse_aggregate, fine_aggregate, i]
        final_features = normalize([f_list])
        concrete_data = pd.DataFrame(final_features)

        # Predict the compressive strength of concrete using the trained model
        predicted = model.predict(concrete_data)
        predicted = [float(np.round(x)) for x in predicted][0]

        # Calculate the result using the formula
        # Strength = 2.12 * (cement + blast_furnace_slag + fly_ash + water + superplasticizer + coarse_aggregate + fine_aggregate) + 0.001 * age
        calculated =  2.12 * (cement + blast_furnace_slag + fly_ash + water + superplasticizer + coarse_aggregate + fine_aggregate) + 0.001 * i
    
        results.append({
            'predicted': round(predicted, 2),
            'calculated': round(calculated, 2),
            'age': i
        })

    # Return the data
    return {'results': results}

if __name__ == '__main__':
    # Just an example of how to set a secret key
    app.run(debug = True)
