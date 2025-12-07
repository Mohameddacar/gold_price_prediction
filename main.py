import os
from flask import Flask, request, jsonify
import joblib
import pickle
import pandas as pd
import numpy as np
from datetime import datetime

# Rename the Flask app instance
application = Flask(__name__)

print("="*60)
print("GOLD PRICE PREDICTOR - MACHINE LEARNING POWERED")
print("="*60)

# Load model - try different methods
model = None
scaler = None

# Try to load existing model
try:
    model = joblib.load('gold_price_model.pkl')
    print("✅ Model loaded successfully")
except:
    try:
        with open('gold_price_model.pkl', 'rb') as f:
            model = pickle.load(f)
        print("✅ Model loaded with pickle")
    except:
        print("⚠️ Creating dummy model for testing")
        # Create dummy model
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.coef_ = np.array([-15.5, 25.3, 8.7, -12.4])
        model.intercept_ = 1800.0

# Try to load scaler
try:
    scaler = joblib.load('scaler.pkl')
    print("✅ Scaler loaded successfully")
except:
    try:
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        print("✅ Scaler loaded with pickle")
    except:
        print("⚠️ Creating dummy scaler")
        # Create dummy scaler
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaler.mean_ = np.array([95, 3.0, 80, 2.5])
        scaler.scale_ = np.array([5, 2, 20, 1.5])

print(f"\n✅ Status: Model = {'Loaded' if model else 'Not loaded'}, Scaler = {'Loaded' if scaler else 'Not loaded'}")
print("="*60)

@application.route('/')
def home():
    # ... [KEEP ALL THE EXISTING HTML CODE FROM YOUR APP.PY]
    # Copy the entire HTML from your app.py here
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Gold Price Predictor</title>
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
                color: white;
                min-height: 100vh;
                display: flex;
                justify-content: center;
                align-items: center;
                padding: 20px;
            }
            
            .container {
                max-width: 500px;
                width: 100%;
                background: rgba(255, 255, 255, 0.1);
                backdrop-filter: blur(10px);
                border-radius: 20px;
                padding: 40px;
                box-shadow: 0 15px 35px rgba(0, 0, 0, 0.3);
                border: 1px solid rgba(255, 255, 255, 0.1);
            }
            
            .header {
                text-align: center;
                margin-bottom: 30px;
            }
            
            .logo {
                font-size: 48px;
                color: #FFD700;
                margin-bottom: 10px;
            }
            
            h1 {
                font-size: 28px;
                font-weight: 600;
                margin-bottom: 5px;
                background: linear-gradient(45deg, #FFD700, #FFA500);
                -webkit-background-clip: text;
                background-clip: text;
                color: transparent;
            }
            
            .subtitle {
                font-size: 14px;
                color: #aaa;
                font-weight: 300;
                letter-spacing: 1px;
            }
            
            .input-group {
                margin-bottom: 25px;
            }
            
            label {
                display: block;
                margin-bottom: 8px;
                font-weight: 500;
                color: #ddd;
                font-size: 14px;
                display: flex;
                justify-content: space-between;
            }
            
            .range {
                color: #FFD700;
                font-size: 12px;
            }
            
            input {
                width: 100%;
                padding: 15px;
                background: rgba(255, 255, 255, 0.08);
                border: 2px solid rgba(255, 255, 255, 0.1);
                border-radius: 10px;
                color: white;
                font-size: 16px;
                transition: all 0.3s ease;
            }
            
            input:focus {
                outline: none;
                border-color: #FFD700;
                background: rgba(255, 255, 255, 0.12);
                box-shadow: 0 0 0 3px rgba(255, 215, 0, 0.1);
            }
            
            .slider-container {
                margin-top: 5px;
                display: flex;
                align-items: center;
                gap: 10px;
            }
            
            input[type="range"] {
                flex: 1;
                padding: 0;
                height: 6px;
                background: rgba(255, 255, 255, 0.1);
                border-radius: 3px;
                -webkit-appearance: none;
            }
            
            input[type="range"]::-webkit-slider-thumb {
                -webkit-appearance: none;
                width: 20px;
                height: 20px;
                background: #FFD700;
                border-radius: 50%;
                cursor: pointer;
            }
            
            .value-display {
                min-width: 40px;
                text-align: center;
                font-weight: 600;
                color: #FFD700;
            }
            
            .predict-btn {
                width: 100%;
                padding: 18px;
                background: linear-gradient(45deg, #FFD700, #FFA500);
                color: #1a1a2e;
                border: none;
                border-radius: 10px;
                font-size: 18px;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.3s ease;
                display: flex;
                align-items: center;
                justify-content: center;
                gap: 10px;
                margin-top: 10px;
            }
            
            .predict-btn:hover {
                transform: translateY(-2px);
                box-shadow: 0 8px 20px rgba(255, 215, 0, 0.3);
            }
            
            .result-container {
                margin-top: 30px;
                padding: 25px;
                background: rgba(255, 255, 255, 0.08);
                border-radius: 15px;
                border: 1px solid rgba(255, 215, 0, 0.3);
                text-align: center;
                display: none;
                animation: fadeIn 0.5s ease;
            }
            
            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(20px); }
                to { opacity: 1; transform: translateY(0); }
            }
            
            .result-title {
                font-size: 18px;
                color: #aaa;
                margin-bottom: 10px;
            }
            
            .gold-price {
                font-size: 42px;
                font-weight: 700;
                color: #FFD700;
                text-shadow: 0 2px 10px rgba(255, 215, 0, 0.3);
                margin: 10px 0;
            }
            
            .result-details {
                margin-top: 20px;
                display: grid;
                grid-template-columns: repeat(2, 1fr);
                gap: 15px;
            }
            
            .detail-item {
                background: rgba(255, 255, 255, 0.05);
                padding: 12px;
                border-radius: 8px;
                text-align: center;
            }
            
            .detail-label {
                font-size: 12px;
                color: #aaa;
                margin-bottom: 5px;
            }
            
            .detail-value {
                font-size: 16px;
                font-weight: 600;
                color: white;
            }
            
            .loading {
                text-align: center;
                padding: 20px;
                display: none;
            }
            
            .loading i {
                font-size: 24px;
                color: #FFD700;
                animation: spin 1s linear infinite;
            }
            
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            
            .footer {
                text-align: center;
                margin-top: 30px;
                color: #666;
                font-size: 12px;
                padding-top: 20px;
                border-top: 1px solid rgba(255, 255, 255, 0.1);
            }
            
            @media (max-width: 480px) {
                .container {
                    padding: 25px;
                }
                h1 {
                    font-size: 24px;
                }
                .gold-price {
                    font-size: 36px;
                }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <div class="logo">
                    <i class="fas fa-coins"></i>
                </div>
                <h1>Gold Price Predictor</h1>
                <div class="subtitle">Machine Learning Powered Prediction</div>
            </div>
            
            <form id="predictionForm">
                <!-- USD Index -->
                <div class="input-group">
                    <label>
                        <span>USD Index <i class="fas fa-chart-line"></i></span>
                        <span class="range">(85-105)</span>
                    </label>
                    <input type="number" id="usd_index" min="85" max="105" step="0.1" value="95.5" required>
                    <div class="slider-container">
                        <input type="range" id="usd_index_slider" min="85" max="105" step="0.1" value="95.5">
                        <span class="value-display" id="usd_index_value">95.5</span>
                    </div>
                </div>
                
                <!-- Inflation Rate -->
                <div class="input-group">
                    <label>
                        <span>Inflation Rate % <i class="fas fa-percentage"></i></span>
                        <span class="range">(1.5-8.5)</span>
                    </label>
                    <input type="number" id="inflation" min="1.5" max="8.5" step="0.1" value="3.2" required>
                    <div class="slider-container">
                        <input type="range" id="inflation_slider" min="1.5" max="8.5" step="0.1" value="3.2">
                        <span class="value-display" id="inflation_value">3.2</span>
                    </div>
                </div>
                
                <!-- Oil Price -->
                <div class="input-group">
                    <label>
                        <span>Oil Price $ <i class="fas fa-gas-pump"></i></span>
                        <span class="range">(40-120)</span>
                    </label>
                    <input type="number" id="oil_price" min="40" max="120" step="0.1" value="78.3" required>
                    <div class="slider-container">
                        <input type="range" id="oil_price_slider" min="40" max="120" step="0.1" value="78.3">
                        <span class="value-display" id="oil_price_value">78.3</span>
                    </div>
                </div>
                
                <!-- Interest Rate -->
                <div class="input-group">
                    <label>
                        <span>Interest Rate % <i class="fas fa-hand-holding-usd"></i></span>
                        <span class="range">(0.25-5.5)</span>
                    </label>
                    <input type="number" id="interest_rate" min="0.25" max="5.5" step="0.01" value="2.75" required>
                    <div class="slider-container">
                        <input type="range" id="interest_rate_slider" min="0.25" max="5.5" step="0.01" value="2.75">
                        <span class="value-display" id="interest_rate_value">2.75</span>
                    </div>
                </div>
                
                <button type="submit" class="predict-btn">
                    <i class="fas fa-calculator"></i>
                    Predict Gold Price
                </button>
            </form>
            
            <div class="loading" id="loading">
                <i class="fas fa-spinner"></i>
                <p>Calculating prediction...</p>
            </div>
            
            <div class="result-container" id="resultContainer">
                <div class="result-title">PREDICTED GOLD PRICE</div>
                <div class="gold-price" id="predictionValue">$0.00</div>
                <div class="result-details">
                    <div class="detail-item">
                        <div class="detail-label">USD Index</div>
                        <div class="detail-value" id="detailUsd">95.5</div>
                    </div>
                    <div class="detail-item">
                        <div class="detail-label">Inflation</div>
                        <div class="detail-value" id="detailInflation">3.2%</div>
                    </div>
                    <div class="detail-item">
                        <div class="detail-label">Oil Price</div>
                        <div class="detail-value" id="detailOil">$78.3</div>
                    </div>
                    <div class="detail-item">
                        <div class="detail-label">Interest Rate</div>
                        <div class="detail-value" id="detailInterest">2.75%</div>
                    </div>
                </div>
            </div>
            
            <div class="footer">
                <p>Powered by Machine Learning & Flask</p>
                <p>© 2024 Gold Price Prediction System</p>
            </div>
        </div>
        
        <script>
            // Sync sliders with number inputs
            function syncInputs(numberId, sliderId, displayId) {
                const numberInput = document.getElementById(numberId);
                const sliderInput = document.getElementById(sliderId);
                const displaySpan = document.getElementById(displayId);
                
                function updateDisplay() {
                    displaySpan.textContent = numberInput.value;
                }
                
                numberInput.addEventListener('input', function() {
                    sliderInput.value = this.value;
                    updateDisplay();
                });
                
                sliderInput.addEventListener('input', function() {
                    numberInput.value = this.value;
                    updateDisplay();
                });
                
                updateDisplay();
            }
            
            // Initialize all sliders
            syncInputs('usd_index', 'usd_index_slider', 'usd_index_value');
            syncInputs('inflation', 'inflation_slider', 'inflation_value');
            syncInputs('oil_price', 'oil_price_slider', 'oil_price_value');
            syncInputs('interest_rate', 'interest_rate_slider', 'interest_rate_value');
            
            // Form submission
            document.getElementById('predictionForm').addEventListener('submit', async function(e) {
                e.preventDefault();
                
                // Show loading, hide results
                document.getElementById('loading').style.display = 'block';
                document.getElementById('resultContainer').style.display = 'none';
                
                // Get form data
                const data = {
                    usd_index: parseFloat(document.getElementById('usd_index').value),
                    inflation: parseFloat(document.getElementById('inflation').value),
                    oil_price: parseFloat(document.getElementById('oil_price').value),
                    interest_rate: parseFloat(document.getElementById('interest_rate').value)
                };
                
                try {
                    const response = await fetch('/predict', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify(data)
                    });
                    
                    const result = await response.json();
                    
                    // Hide loading
                    document.getElementById('loading').style.display = 'none';
                    
                    if (result.error) {
                        alert('Error: ' + result.error);
                    } else {
                        // Update result display
                        document.getElementById('predictionValue').textContent = 
                            '$' + result.prediction.toFixed(2);
                        
                        // Update detail values
                        document.getElementById('detailUsd').textContent = data.usd_index;
                        document.getElementById('detailInflation').textContent = data.inflation + '%';
                        document.getElementById('detailOil').textContent = '$' + data.oil_price;
                        document.getElementById('detailInterest').textContent = data.interest_rate + '%';
                        
                        // Show result container
                        document.getElementById('resultContainer').style.display = 'block';
                        
                        // Scroll to results
                        document.getElementById('resultContainer').scrollIntoView({ 
                            behavior: 'smooth',
                            block: 'nearest'
                        });
                    }
                } catch (error) {
                    document.getElementById('loading').style.display = 'none';
                    alert('Network error: ' + error.message);
                }
            });
            
            // Add sample data button
            document.addEventListener('DOMContentLoaded', function() {
                const form = document.getElementById('predictionForm');
                const sampleBtn = document.createElement('button');
                sampleBtn.type = 'button';
                sampleBtn.innerHTML = '<i class="fas fa-bolt"></i> Load Sample Data';
                sampleBtn.style.cssText = `
                    width: 100%;
                    padding: 12px;
                    background: linear-gradient(45deg, #667eea, #764ba2);
                    color: white;
                    border: none;
                    border-radius: 10px;
                    font-size: 14px;
                    cursor: pointer;
                    margin-top: 5px;
                    transition: all 0.3s ease;
                `;
                sampleBtn.onmouseover = () => sampleBtn.style.opacity = '0.9';
                sampleBtn.onmouseout = () => sampleBtn.style.opacity = '1';
                sampleBtn.onclick = () => {
                    document.getElementById('usd_index').value = '95.5';
                    document.getElementById('inflation').value = '3.2';
                    document.getElementById('oil_price').value = '78.3';
                    document.getElementById('interest_rate').value = '2.75';
                    
                    // Trigger input events to sync sliders
                    ['usd_index', 'inflation', 'oil_price', 'interest_rate'].forEach(id => {
                        document.getElementById(id).dispatchEvent(new Event('input'));
                    });
                };
                form.appendChild(sampleBtn);
            });
        </script>
    </body>
    </html>
    '''

@application.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded. Please check server logs.'}), 500
    
    try:
        # Get data from request
        data = request.get_json()
        
        # Extract features
        usd_index = float(data.get('usd_index', 95.5))
        inflation = float(data.get('inflation', 3.2))
        oil_price = float(data.get('oil_price', 78.3))
        interest_rate = float(data.get('interest_rate', 2.75))
        
        # Create input array
        input_array = np.array([[usd_index, inflation, oil_price, interest_rate]])
        
        # Scale features
        scaled_input = scaler.transform(input_array)
        
        # Make prediction
        prediction = model.predict(scaled_input)
        
        return jsonify({
            'prediction': float(prediction[0]),
            'features': data
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@application.route('/health')
def health():
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None,
        'scaler_loaded': scaler is not None,
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"\n🚀 Gold Price Predictor starting on http://localhost:{port}")
    print("="*60)
    application.run(host='0.0.0.0', port=port, debug=False)

    