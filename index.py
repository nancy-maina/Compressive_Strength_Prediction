import numpy as np
import tensorflow as tf

from flask import Flask, render_template, request
from sklearn.preprocessing import normalize, MinMaxScaler

app = Flask(__name__)

# static variable that should be loaded only once
model = tf.keras.models.load_model('model_data/my_model.h5', compile= False)
model.load_weights('model_data/my_model_weights.h5')
model.compile()

# Validate inputs from the form
valid_inputs = {
    'cement': (0, 1000),
    'blast_furnace_slag': (0, 5),
    'fly_ash': (0, 5),
    'water': (0, 500),
    'superplasticizer': (0, 5),
    'coarse_aggregate': (0, 4000),
    'fine_aggregate': (0, 2000),
    'age': (0, 365)
}

def predict_concrete_strength(cement, blast_furnace_slag, fly_ash, water, superplasticizer, coarse_aggregate, fine_aggregate, age):
    # [cement, blast_furnace_slag, fly_ash, water, superplasticizer, coarse_aggregate, fine_aggregate, age]
    final_features = normalize([[cement, blast_furnace_slag, fly_ash, water, superplasticizer, coarse_aggregate, fine_aggregate, age]])
    prediction = model.predict(final_features)
    return [float(np.round(x)) for x in prediction][0]

@app.route('/')
def index():
    return render_template('index.html', errors={}, form={})

@app.route('/predict', methods=['POST'])
def predict():
    # Validate inputs from the form
    # cement, blast_furnace_slag, fly_ash, water, superplasticizer, coarse_aggregate, fine_aggregate, age
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

    # Make a prediction
    predicted = predict_concrete_strength(cement, blast_furnace_slag, fly_ash, water, superplasticizer, coarse_aggregate, fine_aggregate, age)

    # Calculate the difference between the two results, i.e. the error, accuracy, mean absolute error, etc.
    return render_template('prediction.html', predicted=predicted)

@app.route('/chart')
def chart():
    return render_template('chart.html')


# Chart data
@app.route('/chart_data', methods=['GET'])
def chart_data():
    # Validate inputs from the form
    # cement, blast_furnace_slag, fly_ash, water, superplasticizer, coarse_aggregate, fine_aggregate
    errors = {}
    for key, value in valid_inputs.items():
        if key == 'age':
            continue

        if not request.args.get(key) or not value[0] <= float(request.args.get(key)) <= value[1]:
            errors[key] = f"Please enter a valid value for {key}. Valid values are between {value[0]} and {value[1]}."

    if errors:
        # Set the status code to 400 and return the errors
        return {'errors': errors}, 422

    # Generate some data given the inputs
    cement = float(request.args.get('cement'))
    blast_furnace_slag = float(request.args.get('blast_furnace_slag'))
    fly_ash = float(request.args.get('fly_ash'))
    water = float(request.args.get('water'))
    superplasticizer = float(request.args.get('superplasticizer'))
    coarse_aggregate = float(request.args.get('coarse_aggregate'))
    fine_aggregate = float(request.args.get('fine_aggregate'))
    
    # Get results for the next 28 days
    results = []

    for i in range(29):
        # Make a prediction
        predicted = predict_concrete_strength(cement, blast_furnace_slag, fly_ash, water, superplasticizer, coarse_aggregate, fine_aggregate, i)
    
        results.append({
            'predicted': round(predicted, 2),
            'age': i
        })

    # Return the data
    return {'results': results}

if __name__ == '__main__':
    # Just an example of how to set a secret key
    app.run(debug = True)
