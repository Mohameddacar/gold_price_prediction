import os
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import numpy as np
from datetime import datetime

app = Flask(__name__)

# Load model and scaler
try:
    model = pickle.load(open('gold_price_model.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))
    print("Model and scaler loaded successfully")
except Exception as e:
    print(f"Error loading model/scaler: {e}")
    model = None
    scaler = None

@app.route('/')
def home():
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Gold Price Prediction</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .container { max-width: 600px; margin: 0 auto; }
            .form-group { margin-bottom: 15px; }
            label { display: block; margin-bottom: 5px; }
            input { width: 100%; padding: 8px; box-sizing: border-box; }
            button { background: #4CAF50; color: white; padding: 10px 20px; border: none; cursor: pointer; }
            .result { margin-top: 20px; padding: 15px; background: #f4f4f4; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Gold Price Prediction</h1>
            <form id="predictionForm">
                <div class="form-group">
                    <label for="spx">SPX Index:</label>
                    <input type="number" step="any" id="spx" name="spx" required>
                </div>
                <div class="form-group">
                    <label for="uso">USO Price:</label>
                    <input type="number" step="any" id="uso" name="uso" required>
                </div>
                <div class="form-group">
                    <label for="slv">SLV Price:</label>
                    <input type="number" step="any" id="slv" name="slv" required>
                </div>
                <div class="form-group">
                    <label for="eur_usd">EUR/USD Rate:</label>
                    <input type="number" step="any" id="eur_usd" name="eur_usd" required>
                </div>
                <button type="submit">Predict Gold Price</button>
            </form>
            <div id="result" class="result" style="display:none;"></div>
        </div>
        
        <script>
            document.getElementById('predictionForm').addEventListener('submit', async function(e) {
                e.preventDefault();
                
                const formData = {
                    spx: parseFloat(document.getElementById('spx').value),
                    uso: parseFloat(document.getElementById('uso').value),
                    slv: parseFloat(document.getElementById('slv').value),
                    eur_usd: parseFloat(document.getElementById('eur_usd').value)
                };
                
                try {
                    const response = await fetch('/predict', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify(formData)
                    });
                    
                    const result = await response.json();
                    const resultDiv = document.getElementById('result');
                    
                    if (result.error) {
                        resultDiv.innerHTML = `<h3>Error:</h3><p>${result.error}</p>`;
                    } else {
                        resultDiv.innerHTML = `
                            <h3>Prediction Result:</h3>
                            <p><strong>Predicted Gold Price:</strong> $${result.prediction.toFixed(2)}</p>
                            <p><strong>Features Used:</strong></p>
                            <ul>
                                <li>SPX: ${result.features.spx}</li>
                                <li>USO: ${result.features.uso}</li>
                                <li>SLV: ${result.features.slv}</li>
                                <li>EUR/USD: ${result.features.eur_usd}</li>
                            </ul>
                        `;
                    }
                    resultDiv.style.display = 'block';
                } catch (error) {
                    document.getElementById('result').innerHTML = 
                        `<h3>Error:</h3><p>${error.message}</p>`;
                    document.getElementById('result').style.display = 'block';
                }
            });
        </script>
    </body>
    </html>
    '''

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or scaler is None:
        return jsonify({'error': 'Model not loaded properly'}), 500
    
    try:
        # Get data from request
        data = request.get_json()
        
        # Extract features (adjust based on your model's expected features)
        features = [
            float(data.get('spx', 0)),
            float(data.get('uso', 0)),
            float(data.get('slv', 0)),
            float(data.get('eur_usd', 0))
        ]
        
        # Create DataFrame with correct column names
        # ADJUST THESE COLUMN NAMES BASED ON YOUR MODEL
        feature_names = ['SPX', 'USO', 'SLV', 'EUR/USD']
        input_df = pd.DataFrame([features], columns=feature_names)
        
        # Scale features if scaler was used during training
        if scaler:
            scaled_features = scaler.transform(input_df)
        else:
            scaled_features = input_df.values
        
        # Make prediction
        prediction = model.predict(scaled_features)
        
        # Return result
        return jsonify({
            'prediction': float(prediction[0]),
            'features': data
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'scaler_loaded': scaler is not None,
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)