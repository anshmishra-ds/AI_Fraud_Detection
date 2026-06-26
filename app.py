from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Train kiya hua AI model load karna
with open('model/fraud_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Form se data lena
        amount = float(request.form['amount'])
        international = int(request.form['international'])
        midnight = int(request.form['midnight'])
        
        # AI Model ke format me input arrange karna
        input_features = np.array([[amount, international, midnight]])
        
        # Prediction karna
        prediction = model.predict(input_features)
        
        if prediction[0] == 1:
            result = "⚠️ ALERT: This transaction looks like a FRAUD!"
        else:
            result = "✅ SAFE: This transaction is safe."
            
        return render_template('index.html', prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)
