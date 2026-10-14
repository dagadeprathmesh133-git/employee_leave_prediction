import pickle
import numpy as np
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Load the Naive Bayes model
with open('naive_bayes_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract features matching the model's training columns[cite: 1]
        # Categorical mapping dictionaries based on standard label encoding
        education_map = {'Bachelors': 0, 'Masters': 1, 'PHD': 2}
        city_map = {'Bangalore': 0, 'New Delhi': 1, 'Pune': 2}
        gender_map = {'Female': 0, 'Male': 1}
        benched_map = {'No': 0, 'Yes': 1}

        education = education_map.get(request.form.get('Education'), 0)
        joining_year = float(request.form.get('JoiningYear', 2018))
        city = city_map.get(request.form.get('City'), 0)
        payment_tier = int(request.form.get('PaymentTier', 1))
        age = float(request.form.get('Age', 25))
        gender = gender_map.get(request.form.get('Gender'), 0)
        ever_benched = benched_map.get(request.form.get('EverBenched'), 0)
        experience = float(request.form.get('ExperienceInCurrentDomain', 1))

        # Assemble feature array in the exact order required by the model[cite: 1]
        features = np.array([[
            education, 
            joining_year, 
            city, 
            payment_tier, 
            age, 
            gender, 
            ever_benched, 
            experience
        ]])

        # Make prediction
        prediction = model.predict(features)
        probability = model.predict_proba(features)

        result_text = "Likely to Leave / Positive Class" if prediction[0] == 1 else "Likely to Stay / Negative Class"
        confidence = round(np.max(probability) * 100, 2)

        return render_template('index.html', 
                               prediction_text=f"Prediction: {result_text}", 
                               confidence_text=f"Confidence: {confidence}%")
    
    except Exception as e:
        return render_template('index.html', prediction_text=f"Error occurred: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
