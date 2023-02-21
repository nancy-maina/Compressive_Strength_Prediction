from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf

app = Flask(__name__)

model =pickle.load(open('model.pkl', 'rb'))
@app.route('/')
def hello_world():
    return render_template('test.html')

@app.route('/predict', methods= ['POST', 'GET'])
def predict():
    if request.method == "POST":
        # ['age', 'cement', 'water', 'fly_ash', 'superplasticizer', 'blast_furnace_slag']
        f_list = [request.form.get('age'), request.form.get('cement'), request.form.get('water'),
                  request.form.get('fa'),
                  request.form.get('sp'), request.form.get('bfs')]  # list of inputs
        
         # logging operation
#         logging.info(f"Age (in days): {f_list[0]}, Cement (in kg): {f_list[1]},"
#                      f"Water (in kg): {f_list[2]}, Fly ash (in kg): {f_list[3]},"
#                      f"Superplasticizer (in kg): {f_list[4]}, Blast furnace slag (in kg): {f_list[5]}")

        final_features = np.array(f_list).reshape(-1, 8)
        concrete_data = pd.DataFrame(final_features)

        prediction = model.predict(concrete_data)
        result = "%.2f" % round(prediction[0], 2)

        # logging operation
#         logging.info(f"The Predicted Concrete Compressive strength is {result} MPa")

#         logging.info("Prediction getting posted to the web page.")
        return render_template('test.html',
                               prediction_text=f"The Concrete compressive strength is {result} MPa")
    


if __name__ == '__main__':
    app.run()