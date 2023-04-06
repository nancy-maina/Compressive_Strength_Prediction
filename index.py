import numpy as np
import pandas as pd
import pickle
import tensorflow as tf

from flask import Flask, render_template, request

app = Flask(__name__)

model = tf.keras.models.load_model('my_model_weights.h5')


@app.route('/')
def hello_world():
    return render_template('test.html')


@app.route('/predict', methods=['POST'])
def predict():
    # [cement	bfs	fa	water	sp	coarse_aggregate	fine_aggregate	age	]
    f_list = [
        float(request.form.get('cement')),
        float(request.form.get('bfs')),
        float(request.form.get('fa')),
        float(request.form.get('water')),
        float(request.form.get('sp')),
        float(request.form.get('coarse_aggregate')),
        float(request.form.get('fine_aggregate')),
        float(request.form.get('age'))
    ]  # list of inputs

    # logging operation
#         logging.info(f"Age (in days): {f_list[0]}, Cement (in kg): {f_list[1]},"
#                      f"Water (in kg): {f_list[2]}, Fly ash (in kg): {f_list[3]},"
#                      f"Superplasticizer (in kg): {f_list[4]}, Blast furnace slag (in kg): {f_list[5]}")

    final_features = np.array(f_list).reshape(-1, 8)
    concrete_data = pd.DataFrame(final_features)

    prediction = model.predict(concrete_data)
    result = [float(np.round(x)) for x in prediction]

    # logging operation
#         logging.info(f"The Predicted Concrete Compressive strength is {result} MPa")

#         logging.info("Prediction getting posted to the web page.")
    return render_template('test.html',
                           prediction_text=f"The Concrete compressive strength is {result} MPa")


if __name__ == '__main__':
    app.run()
