
from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')

app = Flask(__name__)

# Global variables to store model and feature names
model_data = None

def load_model():
    """Load the trained model"""
    global model_data
    try:
        if os.path.exists("model.pkl"):
            model_data = joblib.load("model.pkl")
            print("Model loaded successfully!")
            print(f"Feature names: {model_data['feature_names']}")
            return True
        else:
            print("Model file not found. Training new model...")
            return train_new_model()
    except Exception as e:
        print(f"Error loading model: {e}")
        return train_new_model()

def train_new_model():
    """Train a new model if the saved model doesn't exist"""
    global model_data
    try:
        # Check if CSV file exists
        if not os.path.exists("newborn_health_monitoring_with_risk.csv"):
            print("CSV file not found!")
            return False
        
        # Load and preprocess data
        df = pd.read_csv("newborn_health_monitoring_with_risk.csv")
        
        # Fill missing values in apgar_score
        if 'apgar_score' in df.columns and 'baby_id' in df.columns:
            df['apgar_score'] = df.groupby('baby_id')['apgar_score'].transform(
                lambda g: g.ffill().bfill()
            )
        
        # Drop unnecessary columns
        columns_to_drop = ['baby_id', 'name', 'date']
        for col in columns_to_drop:
            if col in df.columns:
                df.drop(col, axis=1, inplace=True)
        
        # Separate features and target
        X = df.drop('risk_level', axis=1)
        y = df['risk_level']
        
        # Get categorical columns (excluding target)
        categorical_columns = X.select_dtypes(include=['object']).columns.tolist()
        
        # Apply get_dummies to features
        X_encoded = pd.get_dummies(X, drop_first=True)
        
        # For target variable, we need to check if it's binary or multiclass
        if len(y.unique()) == 2:
            # Binary classification - use LabelEncoder to ensure proper encoding
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)
        else:
            # Multi-class - use get_dummies
            y_encoded = pd.get_dummies(y, drop_first=True)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X_encoded, y_encoded, test_size=0.2, stratify=y_encoded, random_state=5
        )
        
        # Train Decision Tree model
        tree_clf = DecisionTreeClassifier(random_state=42)
        tree_clf.fit(X_train, y_train)
        
        # Create encoding information for categorical variables
        categorical_encoding = {}
        for col in categorical_columns:
            categorical_encoding[col] = {}
            unique_values = X[col].unique()
            for i, val in enumerate(unique_values):
                categorical_encoding[col][val] = i
        
        # Save model data
        model_data = {
            'model': tree_clf,
            'feature_names': list(X_encoded.columns),
            'categorical_encoding': categorical_encoding,
            'original_categorical_columns': categorical_columns
        }
        
        # Save to file
        joblib.dump(model_data, "model.pkl")
        print("New model trained and saved successfully!")
        return True
        
    except Exception as e:
        print(f"Error training new model: {e}")
        return False

def preprocess_input(data):
    """Preprocess input data to match the training format"""
    try:
        # Create DataFrame from input data
        df = pd.DataFrame([data])
        
        # Apply get_dummies with the same structure as training
        df_encoded = pd.get_dummies(df, drop_first=True)
        
        # Ensure all required features are present
        missing_features = set(model_data['feature_names']) - set(df_encoded.columns)
        for feature in missing_features:
            df_encoded[feature] = 0
        
        # Ensure columns are in the same order as training
        df_encoded = df_encoded.reindex(columns=model_data['feature_names'], fill_value=0)
        
        return df_encoded.values
        
    except Exception as e:
        print(f"Error in preprocessing: {e}")
        raise e

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Make prediction based on input data"""
    try:
        # Get JSON data from request
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Check if model is loaded
        if model_data is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Preprocess the input data
        processed_data = preprocess_input(data)
        
        # Make prediction
        prediction = model_data['model'].predict(processed_data)[0]
        
        # Get prediction probabilities for confidence
        try:
            probabilities = model_data['model'].predict_proba(processed_data)[0]
            confidence = max(probabilities) * 100
        except:
            # If predict_proba is not available, use decision function or default confidence
            confidence = 85.0
        
        # Convert prediction to readable format
        if isinstance(prediction, (int, np.integer)):
            risk_level = "At Risk" if prediction == 1 else "Healthy"
        else:
            # Handle string predictions from the model
            if str(prediction).lower() in ['at risk', 'atrisk', '1']:
                risk_level = "At Risk"
            else:
                risk_level = "Healthy"
        
        return jsonify({
            'risk_level': risk_level,
            'confidence': confidence,
            'raw_prediction': str(prediction)
        })
        
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    status = "OK" if model_data is not None else "Model not loaded"
    return jsonify({
        'status': status,
        'model_loaded': model_data is not None
    })

@app.route('/model-info')
def model_info():
    """Get information about the loaded model"""
    if model_data is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    return jsonify({
        'feature_count': len(model_data['feature_names']),
        'features': model_data['feature_names'],
        'categorical_columns': model_data.get('original_categorical_columns', [])
    })

if __name__ == '__main__':
    # Load model on startup
    if not load_model():
        print("Failed to load or train model. Please check your data files.")
        exit(1)
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)
