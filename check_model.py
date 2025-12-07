import pickle
import joblib
import numpy as np

print("Checking model files...")

# Try to load model
try:
    with open('gold_price_model.pkl', 'rb') as f:
        model = pickle.load(f)
        print("✅ Model loaded successfully!")
        print(f"Model type: {type(model)}")
        print(f"Model attributes: {dir(model)}")
        
        # Check if it's a scikit-learn model
        if hasattr(model, 'predict'):
            print("✅ Model has predict method")
        else:
            print("❌ Model doesn't have predict method")
            
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("Trying with joblib...")
    try:
        model = joblib.load('gold_price_model.pkl')
        print("✅ Model loaded with joblib!")
    except:
        print("❌ Failed with joblib too")

print("\n" + "="*50 + "\n")

# Try to load scaler
try:
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
        print("✅ Scaler loaded successfully!")
        print(f"Scaler type: {type(scaler)}")
        
        # Check scaler attributes
        if hasattr(scaler, 'transform'):
            print("✅ Scaler has transform method")
        if hasattr(scaler, 'mean_'):
            print(f"✅ Scaler mean: {scaler.mean_}")
        if hasattr(scaler, 'scale_'):
            print(f"✅ Scaler scale: {scaler.scale_}")
            
except Exception as e:
    print(f"❌ Error loading scaler: {e}")