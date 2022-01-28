from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__) 
model = pickle.load(open('models/logreg.pkl', 'rb'))
scaler_ = pickle.load(open('models/regresi.pkl', 'rb'))


@app.route("/")
def home():
    return render_template("main.html")

@app.route("/predict", methods=['POST'])
def predict():
    maxtemp = float(request.form['maxtemp'])
    rainfall = float(request.form['rainfall'])
    sunshine = float(request.form['sunshine'])
    windspeed = float(request.form['windspeed'])                                                                                                                                                     
    humidity9 = float(request.form['humidity9'])
    humidity3 = float(request.form['humidity3'])
    pressure = float(request.form['pressure'])
    cloud = float(request.form['cloud'])
    month = int(request.form['month'])
    raintoday = int(request.form['raintoday'])
    location = int(request.form['location'])

    val = [maxtemp, rainfall, sunshine, windspeed, humidity9, humidity3, pressure, cloud, month]
    val = scaler_.transform([val])
    val = val.reshape(9,)


    if raintoday == 1:
        val = np.append(val, 1)
    elif raintoday == 0:
        val = np.append(val, 0)
    else:
        print('ERROR!')

    locations = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5,
                6:6, 7:7, 8:8, 9:9, 10:10, 11:11,
                12:12, 13:13, 14:14, 15:15, 16:16, 17:17,
                18:18, 19:19, 20:20, 21:21, 22:22, 23:23,
                24:24, 25:25, 26:26, 27:27, 28:28, 29:29,
                30:30, 31:31, 32:32, 33:33, 34:34, 35:35,
                36:36, 37:37, 38:38, 39:39, 40:40, 41:41, 
                42:42, 43:43, 44:44, 45:45, 46:46}

    for i in range(0,47):
        if locations[location]==i:
            val = np.append(val, 1)
        else:
            val = np.append(val, 0)

    print(val)

    val_predict = model.predict([val])

    if val_predict == 1:
        output = 'Prediksi bernilai 1, maka akan diprediksi besok turun hujan'
    elif val_predict == 0:
        output = 'Prediksi bernilai 0, maka akan diprediksi besok TIDAK turun hujan'
    else:
        output = 'Prediksi tidak valid'

    return render_template('main.html', prediction_text='{}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)