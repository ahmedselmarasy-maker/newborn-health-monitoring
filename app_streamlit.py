import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Newborn Health Monitoring System",
    page_icon="👶",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #2c3e50;
        margin-bottom: 2rem;
    }
    .risk-high {
        background-color: #ffebee;
        color: #c62828;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #c62828;
        margin: 1rem 0;
    }
    .risk-low {
        background-color: #e8f5e8;
        color: #2e7d32;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2e7d32;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #dee2e6;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Load or train model
@st.cache_data
def load_model():
    """Load the trained model or train a new one"""
    try:
        if os.path.exists("model.pkl"):
            model_data = joblib.load("model.pkl")
            st.success("✅ Model loaded successfully!")
            return model_data
        else:
            st.warning("⚠️ Model file not found. Training new model...")
            return train_new_model()
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return train_new_model()

def train_new_model():
    """Train a new model if the saved model doesn't exist"""
    try:
        if not os.path.exists("newborn_health_monitoring_with_risk.csv"):
            st.error("❌ CSV file not found!")
            return None
        
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
        
        # Apply get_dummies to features
        X_encoded = pd.get_dummies(X, drop_first=True)
        
        # Encode target variable
        if len(y.unique()) == 2:
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)
        else:
            y_encoded = pd.get_dummies(y, drop_first=True)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X_encoded, y_encoded, test_size=0.2, stratify=y_encoded, random_state=5
        )
        
        # Train Decision Tree model
        tree_clf = DecisionTreeClassifier(random_state=42)
        tree_clf.fit(X_train, y_train)
        
        # Create model data
        model_data = {
            'model': tree_clf,
            'feature_names': list(X_encoded.columns),
            'categorical_encoding': {},
            'original_categorical_columns': X.select_dtypes(include=['object']).columns.tolist()
        }
        
        # Save to file
        joblib.dump(model_data, "model.pkl")
        st.success("✅ New model trained and saved successfully!")
        return model_data
        
    except Exception as e:
        st.error(f"❌ Error training new model: {e}")
        return None

def preprocess_input(data, model_data):
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
        st.error(f"❌ Error in preprocessing: {e}")
        return None

def main():
    # Header
    st.markdown('<h1 class="main-header">👶 Newborn Health Monitoring and Risk Assessment System</h1>', unsafe_allow_html=True)
    
    # Load model
    model_data = load_model()
    
    if model_data is None:
        st.error("❌ Unable to load or train model. Please check your data files.")
        return
    
    # Sidebar for input
    st.sidebar.header("📊 Health Parameters")
    
    # Basic Newborn Data
    st.sidebar.subheader("👶 Basic Newborn Data")
    gestational_age = st.sidebar.number_input("Gestational Age (weeks)", min_value=0.0, max_value=50.0, value=38.0, step=0.1)
    birth_weight = st.sidebar.number_input("Birth Weight (kg)", min_value=0.0, max_value=10.0, value=3.2, step=0.01)
    birth_length = st.sidebar.number_input("Birth Length (cm)", min_value=0.0, max_value=100.0, value=50.0, step=0.1)
    birth_head_circumference = st.sidebar.number_input("Birth Head Circumference (cm)", min_value=0.0, max_value=50.0, value=35.0, step=0.1)
    gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
    
    # Daily Data
    st.sidebar.subheader("📅 Daily Data")
    age_days = st.sidebar.number_input("Age (days)", min_value=0, max_value=365, value=7)
    current_weight = st.sidebar.number_input("Current Weight (kg)", min_value=0.0, max_value=15.0, value=3.3, step=0.01)
    current_length = st.sidebar.number_input("Current Length (cm)", min_value=0.0, max_value=120.0, value=51.0, step=0.1)
    current_head_circumference = st.sidebar.number_input("Current Head Circumference (cm)", min_value=0.0, max_value=60.0, value=36.0, step=0.1)
    
    # Vital Signs
    st.sidebar.subheader("💓 Vital Signs")
    temperature = st.sidebar.number_input("Temperature (°C)", min_value=30.0, max_value=45.0, value=37.0, step=0.1)
    heart_rate = st.sidebar.number_input("Heart Rate (bpm)", min_value=0, max_value=300, value=120)
    respiratory_rate = st.sidebar.number_input("Respiratory Rate (breaths/min)", min_value=0, max_value=100, value=40)
    oxygen_saturation = st.sidebar.number_input("Oxygen Saturation (%)", min_value=0, max_value=100, value=98)
    
    # Feeding and Output Data
    st.sidebar.subheader("🍼 Feeding and Output Data")
    feeding_type = st.sidebar.selectbox("Feeding Type", ["Breastfeeding", "Formula", "Mixed"])
    feeding_frequency = st.sidebar.number_input("Feeding Frequency (per day)", min_value=0, max_value=20, value=8)
    urine_output = st.sidebar.number_input("Urine Output Count", min_value=0, max_value=20, value=6)
    stool_count = st.sidebar.number_input("Stool Count", min_value=0, max_value=20, value=3)
    
    # Additional Data
    st.sidebar.subheader("🔬 Additional Data")
    jaundice_level = st.sidebar.number_input("Jaundice Level (mg/dl)", min_value=0.0, max_value=50.0, value=8.0, step=0.1)
    apgar_score = st.sidebar.number_input("Apgar Score", min_value=0.0, max_value=10.0, value=8.0, step=0.1)
    immunizations_done = st.sidebar.selectbox("Immunizations Done", ["Yes", "No"])
    reflexes_normal = st.sidebar.selectbox("Reflexes Normal", ["Yes", "No"])
    
    # Collect all data
    input_data = {
        'gestational_age_weeks': gestational_age,
        'birth_weight_kg': birth_weight,
        'birth_length_cm': birth_length,
        'birth_head_circumference_cm': birth_head_circumference,
        'age_days': age_days,
        'weight_kg': current_weight,
        'length_cm': current_length,
        'head_circumference_cm': current_head_circumference,
        'temperature_c': temperature,
        'heart_rate_bpm': heart_rate,
        'respiratory_rate_bpm': respiratory_rate,
        'oxygen_saturation': oxygen_saturation,
        'feeding_frequency_per_day': feeding_frequency,
        'urine_output_count': urine_output,
        'stool_count': stool_count,
        'jaundice_level_mg_dl': jaundice_level,
        'apgar_score': apgar_score,
        'gender': gender,
        'feeding_type': feeding_type,
        'immunizations_done': immunizations_done,
        'reflexes_normal': reflexes_normal
    }
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📋 Current Health Status")
        
        # Display current values
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.metric("Gestational Age", f"{gestational_age} weeks")
            st.metric("Birth Weight", f"{birth_weight} kg")
            st.metric("Current Weight", f"{current_weight} kg")
            st.metric("Temperature", f"{temperature}°C")
            st.metric("Heart Rate", f"{heart_rate} bpm")
        
        with col_b:
            st.metric("Age", f"{age_days} days")
            st.metric("Length", f"{current_length} cm")
            st.metric("Head Circumference", f"{current_head_circumference} cm")
            st.metric("Oxygen Saturation", f"{oxygen_saturation}%")
            st.metric("Apgar Score", f"{apgar_score}")
    
    with col2:
        st.subheader("🎯 Risk Assessment")
        
        # Predict button
        if st.button("🔍 Assess Risk Level", type="primary", use_container_width=True):
            with st.spinner("Analyzing health parameters..."):
                # Preprocess input
                processed_data = preprocess_input(input_data, model_data)
                
                if processed_data is not None:
                    # Make prediction
                    prediction = model_data['model'].predict(processed_data)[0]
                    
                    # Get prediction probabilities
                    try:
                        probabilities = model_data['model'].predict_proba(processed_data)[0]
                        confidence = max(probabilities) * 100
                    except:
                        confidence = 85.0
                    
                    # Convert prediction to readable format
                    if isinstance(prediction, (int, np.integer)):
                        risk_level = "At Risk" if prediction == 1 else "Healthy"
                    else:
                        if str(prediction).lower() in ['at risk', 'atrisk', '1']:
                            risk_level = "At Risk"
                        else:
                            risk_level = "Healthy"
                    
                    # Display result
                    if risk_level == "At Risk":
                        st.markdown(f'<div class="risk-high"><h3>⚠️ Risk Level: {risk_level}</h3><p>Confidence: {confidence:.1f}%</p></div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="risk-low"><h3>✅ Risk Level: {risk_level}</h3><p>Confidence: {confidence:.1f}%</p></div>', unsafe_allow_html=True)
                    
                    # Recommendations
                    st.subheader("💡 Recommendations")
                    if risk_level == "At Risk":
                        st.warning("""
                        **Immediate Actions Required:**
                        - Consult with a pediatrician immediately
                        - Monitor vital signs closely
                        - Ensure proper hydration and nutrition
                        - Schedule follow-up appointments
                        - Consider additional monitoring equipment
                        """)
                    else:
                        st.success("""
                        **Continue Current Care:**
                        - Continue regular monitoring
                        - Maintain current feeding schedule
                        - Schedule routine check-ups
                        - Keep track of growth milestones
                        """)
    

if __name__ == "__main__":
    main()
