import pickle
import numpy as np
from flask import Flask, render_template_string, request

app = Flask(__name__)

# Load the Naive Bayes model
with open('naive_bayes_model.pkl', 'rb') as f:
    model = pickle.load(f)

# HTML and CSS template combined into a single string
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Employee Prediction Portal</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-gradient: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #311042 100%);
            --glass-bg: rgba(255, 255, 255, 0.05);
            --glass-border: rgba(255, 255, 255, 0.1);
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --accent: #6366f1;
            --accent-hover: #4f46e5;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Inter', sans-serif;
        }

        body {
            background: var(--bg-gradient);
            color: var(--text-primary);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }

        .container {
            width: 100%;
            max-width: 700px;
            background: var(--glass-bg);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid var(--glass-border);
            border-radius: 24px;
            padding: 40px;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4);
        }

        .header {
            text-align: center;
            margin-bottom: 30px;
        }

        .header h1 {
            font-size: 26px;
            font-weight: 700;
            margin-bottom: 8px;
            background: linear-gradient(to right, #818cf8, #c084fc);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .header p {
            color: var(--text-secondary);
            font-size: 14px;
        }

        .form-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }

        .input-group {
            display: flex;
            flex-direction: column;
        }

        .input-group.full-width {
            grid-column: span 2;
        }

        label {
            font-size: 13px;
            font-weight: 500;
            color: var(--text-secondary);
            margin-bottom: 6px;
        }

        input, select {
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid var(--glass-border);
            border-radius: 10px;
            padding: 12px 16px;
            color: var(--text-primary);
            font-size: 14px;
            outline: none;
            transition: all 0.3s ease;
        }

        input:focus, select:focus {
            border-color: var(--accent);
            box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.2);
        }

        select option {
            background: #0f172a;
            color: var(--text-primary);
        }

        button {
            grid-column: span 2;
            background: var(--accent);
            color: white;
            border: none;
            border-radius: 12px;
            padding: 14px;
            font-size: 15px;
            font-weight: 600;
            cursor: pointer;
            transition: background 0.3s ease, transform 0.2s ease;
            margin-top: 10px;
        }

        button:hover {
            background: var(--accent-hover);
            transform: translateY(-1px);
        }

        .result-card {
            margin-top: 25px;
            background: rgba(99, 102, 241, 0.1);
            border: 1px solid rgba(99, 102, 241, 0.3);
            border-radius: 12px;
            padding: 20px;
            text-align: center;
        }

        .result-card h3 {
            font-size: 18px;
            color: #818cf8;
            margin-bottom: 6px;
        }

        .result-card p {
            color: var(--text-secondary);
            font-size: 14px;
        }

        @media (max-width: 600px) {
            .form-grid {
                grid-template-columns: 1fr;
            }
            button {
                grid-column: span 1;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Naive Bayes Predictor</h1>
            <p>Enter employee details to generate model inference</p>
        </div>

        <form action="/predict" method="POST">
            <div class="form-grid">
                <div class="input-group">
                    <label for="Education">Education</label>
                    <select name="Education" id="Education" required>
                        <option value="Bachelors">Bachelors</option>
                        <option value="Masters">Masters</option>
                        <option value="PHD">PHD</option>
                    </select>
                </div>

                <div class="input-group">
                    <label for="JoiningYear">Joining Year</label>
                    <input type="number" name="JoiningYear" id="JoiningYear" placeholder="e.g. 2018" required>
                </div>

                <div class="input-group">
                    <label for="City">City</label>
                    <select name="City" id="City" required>
                        <option value="Bangalore">Bangalore</option>
                        <option value="New Delhi">New Delhi</option>
                        <option value="Pune">Pune</option>
                    </select>
                </div>

                <div class="input-group">
                    <label for="PaymentTier">Payment Tier</label>
                    <select name="PaymentTier" id="PaymentTier" required>
                        <option value="1">Tier 1</option>
                        <option value="2">Tier 2</option>
                        <option value="3">Tier 3</option>
                    </select>
                </div>

                <div class="input-group">
                    <label for="Age">Age</label>
                    <input type="number" name="Age" id="Age" placeholder="e.g. 28" required>
                </div>

                <div class="input-group">
                    <label for="Gender">Gender</label>
                    <select name="Gender" id="Gender" required>
                        <option value="Male">Male</option>
                        <option value="Female">Female</option>
                    </select>
                </div>

                <div class="input-group">
                    <label for="EverBenched">Ever Benched</label>
                    <select name="EverBenched" id="EverBenched" required>
                        <option value="No">No</option>
                        <option value="Yes">Yes</option>
                    </select>
                </div>

                <div class="input-group">
                    <label for="ExperienceInCurrentDomain">Experience in Domain (Years)</label>
                    <input type="number" step="1" name="ExperienceInCurrentDomain" id="ExperienceInCurrentDomain" placeholder="e.g. 3" required>
                </div>

                <button type="submit">Run Prediction</button>
            </div>
        </form>

        {% if prediction_text %}
        <div class="result-card">
            <h3>{{ prediction_text }}</h3>
            <p>{{ confidence_text }}</p>
        </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Mapping categorical text fields to numerical values
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

        # Build feature array matching model expectations[cite: 1]
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

        prediction = model.predict(features)
        probability = model.predict_proba(features)

        result_text = "Likely to Leave / Positive Class" if prediction[0] == 1 else "Likely to Stay / Negative Class"
        confidence = round(np.max(probability) * 100, 2)

        return render_template_string(
            HTML_TEMPLATE, 
            prediction_text=f"Prediction: {result_text}", 
            confidence_text=f"Confidence: {confidence}%"
        )
    
    except Exception as e:
        return render_template_string(
            HTML_TEMPLATE, 
            prediction_text=f"Error occurred: {str(e)}", 
            confidence_text=""
        )

if __name__ == '__main__':
    app.run(debug=True)
