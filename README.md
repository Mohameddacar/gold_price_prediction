---
title: Gold Price Prediction
emoji: 💰
colorFrom: yellow
colorTo: gold
sdk: docker
app_file: app.py
pinned: false
---

# Gold Price Predictor 🏆

A machine learning web application that predicts gold prices based on economic indicators.

## Features
- Predict gold prices using USD Index, Inflation, Oil Price, and Interest Rates
- Interactive sliders and input fields
- Beautiful modern UI with gradient effects
- Real-time predictions
- Sample data loading for quick testing

## How to Use
1. Adjust the four input parameters:
   - **USD Index** (85-105)
   - **Inflation Rate %** (1.5-8.5)
   - **Oil Price $** (40-120)
   - **Interest Rate %** (0.25-5.5)
2. Click "Predict Gold Price"
3. View the predicted gold price instantly

## Model Details
- Built with Scikit-learn
- Uses historical economic data for training
- Feature scaling for improved accuracy

## Files
- `app.py` - Main Flask application
- `gold_price_model.pkl` - Trained ML model
- `scaler.pkl` - Feature scaler
- `requirements.txt` - Python dependencies
- `Dockerfile` - Container configuration

## Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Run locally
python app.py

# Visit http://localhost:5000