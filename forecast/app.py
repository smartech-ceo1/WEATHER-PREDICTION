import pickle
import os
import pandas as pd
from flask import Flask, render_template, request

# Explicitly specify the template folder if it's in the parent directory
app = Flask(__name__, template_folder='../templates')

# Paths to the model and label encoders (adjust based on actual locations)
model_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../weather_forecast_model.pkl')
label_encoders_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../label_encoders.pkl')

# Load the trained model and label encoders
try:
    with open(model_file, 'rb') as f:
        model = pickle.load(f)
    with open(label_encoders_file, 'rb') as f:
        label_encoders = pickle.load(f)
except FileNotFoundError as e:
    print(f"Error loading model or label encoders: {e}")
    model = None
    label_encoders = None

@app.route('/')
def index():
    # Render the main page with no messages initially
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not model or not label_encoders:
        return render_template('index.html', error_message="Model not loaded. Please check the server setup.")

    try:
        # Extract input values from the form (ensure all features are included)
        min_temp = float(request.form['min_temp'])
        max_temp = float(request.form['max_temp'])
        rainfall = float(request.form['rainfall'])
        evaporation = float(request.form['evaporation'])
        sunshine = float(request.form['sunshine'])
        windgustdir = request.form['windgustdir']
        windgustspeed = float(request.form['windgustspeed'])
        winddir9am = request.form['winddir9am']
        windspeed9am = float(request.form['windspeed9am'])
        humidity9am = float(request.form['humidity9am'])
        humidity3pm = float(request.form['humidity3pm'])
        pressure9am = float(request.form['pressure9am'])
        pressure3pm = float(request.form['pressure3pm'])
        cloud9am = float(request.form['cloud9am'])
        cloud3pm = float(request.form['cloud3pm'])
        temp9am = float(request.form['temp9am'])
        temp3pm = float(request.form['temp3pm'])
        rain_today = request.form['rain_today']
        date = request.form['date']
        location = request.form['location']

        # Check for missing or invalid values (Optional: Add more checks as needed)
        if any(v is None or v == '' for v in [min_temp, max_temp, rainfall, evaporation, sunshine, windgustdir, 
                                                windgustspeed, winddir9am, windspeed9am, humidity9am, humidity3pm, 
                                                pressure9am, pressure3pm, cloud9am, cloud3pm, temp9am, temp3pm, 
                                                rain_today, date, location]):
            raise ValueError("Missing input values")

        # Combine all inputs into a DataFrame for prediction
        features = pd.DataFrame([[ 
            min_temp, max_temp, rainfall, evaporation, sunshine, windgustdir, windgustspeed,
            winddir9am, windspeed9am, humidity9am, humidity3pm, pressure9am, pressure3pm,
            cloud9am, cloud3pm, temp9am, temp3pm, rain_today, date, location
        ]], columns=[
            'MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine', 'WindGustDir', 
            'WindGustSpeed', 'WindDir9am', 'WindSpeed9am', 'Humidity9am', 'Humidity3pm', 
            'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am', 'Temp3pm', 
            'RainToday', 'Date', 'Location'
        ])

        # Predict using the model
        prediction = model.predict(features)[0]

        # Decode prediction if label encoders are used
        if 'RainTomorrow' in label_encoders:
            prediction = label_encoders['RainTomorrow'].inverse_transform([prediction])[0]

        prediction_text = f"Prediction: {'Yes' if prediction == 'Yes' else 'No'} (Rain Tomorrow)"

    except ValueError as ve:
        return render_template('index.html', error_message=f"Input error: {str(ve)}")
    except Exception as e:
        return render_template('index.html', error_message=f"Error processing input: {str(e)}")

    return render_template('index.html', prediction_text=prediction_text)

if __name__ == '__main__':
    app.run(debug=True)
