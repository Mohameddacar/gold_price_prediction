import joblib
import pickle
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import pandas as pd

print("="*60)
print("Fixing Model Files")
print("="*60)

# First, try to save the existing model with joblib (since it loads with joblib)
try:
    # Load the existing model with joblib
    existing_model = joblib.load('gold_price_model.pkl')
    print(f"✅ Existing model loaded with joblib: {type(existing_model)}")
    
    # Save it with pickle too
    with open('gold_price_model_new.pkl', 'wb') as f:
        pickle.dump(existing_model, f)
    print("✅ Saved model with pickle as gold_price_model_new.pkl")
    
except Exception as e:
    print(f"❌ Error: {e}")

# Create a new scaler (since the existing one is corrupted)
print("\nCreating new scaler...")
scaler = StandardScaler()

# Try to get data from CSV to fit scaler properly
try:
    df = pd.read_csv('gold_price_clean.csv')
    print(f"✅ Loaded data from CSV: {df.shape}")
    
    # Assuming the first 4 columns are features
    X = df.iloc[:, :4].values
    
    # Fit scaler with real data
    scaler.fit(X)
    print(f"✅ Fitted scaler with real data")
    print(f"   Mean: {scaler.mean_}")
    print(f"   Scale: {scaler.scale_}")
    
except:
    print("⚠️ Using synthetic data for scaler")
    # Create synthetic data
    np.random.seed(42)
    X = np.random.randn(100, 4) * 100 + 500
    scaler.fit(X)
    print(f"✅ Fitted scaler with synthetic data")
    print(f"   Mean: {scaler.mean_}")
    print(f"   Scale: {scaler.scale_}")

# Save the new scaler
with open('scaler_new.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("✅ Saved new scaler as scaler_new.pkl")

# Also save with joblib
joblib.dump(scaler, 'scaler_new.joblib')
print("✅ Saved new scaler as scaler_new.joblib")

print("\n" + "="*60)
print("Files created:")
print("1. gold_price_model_new.pkl (pickle format)")
print("2. scaler_new.pkl (pickle format)")
print("3. scaler_new.joblib (joblib format)")
print("="*60)

# Test the new files
print("\nTesting new files...")
try:
    with open('gold_price_model_new.pkl', 'rb') as f:
        test_model = pickle.load(f)
    print("✅ New model loads with pickle!")
    
    with open('scaler_new.pkl', 'rb') as f:
        test_scaler = pickle.load(f)
    print("✅ New scaler loads with pickle!")
    
    # Test prediction
    test_input = np.array([[1500, 40, 35, 1.2]])
    scaled_input = test_scaler.transform(test_input)
    prediction = test_model.predict(scaled_input)
    print(f"✅ Test prediction: ${prediction[0]:.2f}")
    
except Exception as e:
    print(f"❌ Test failed: {e}")