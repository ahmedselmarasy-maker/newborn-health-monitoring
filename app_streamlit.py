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
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    /* White background styling */
    .stApp {
        background-color: #ffffff;
    }
    
    .main-header {
        text-align: center;
        color: #333;
        margin-bottom: 2rem;
        font-size: 2rem;
        font-weight: bold;
    }
    
    /* Container styling */
    .main-container {
        max-width: 800px;
        margin: 0 auto;
        padding: 0 1rem;
    }
    
    /* Card styling */
    .section-card {
        background-color: #f5f5f5;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border: 1px solid #e0e0e0;
    }
    
    .section-title {
        color: #333;
        font-size: 1.2rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    
    /* Input styling */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > select {
        border: 1px solid #ff4444 !important;
        border-radius: 4px !important;
    }
    
    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus,
    .stSelectbox > div > div > select:focus {
        border-color: #ff4444 !important;
        box-shadow: 0 0 0 1px #ff4444 !important;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(90deg, #4285f4, #9c27b0);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        font-weight: bold;
        width: 100%;
        margin-top: 2rem;
    }
    
    .stButton > button:hover {
        background: linear-gradient(90deg, #3367d6, #7b1fa2);
    }
    
    /* Risk result styling */
    .risk-high {
        background-color: #ffebee;
        color: #c62828;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #c62828;
        margin: 1rem 0;
    }
    
    .risk-low {
        background-color: #e8f5e8;
        color: #2e7d32;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2e7d32;
        margin: 1rem 0;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Load or train model
@st.cache_data
def load_model():
    """Load the trained model or train a new one"""
    try:
        if os.path.exists("model.pkl"):
            model_data = joblib.load("model.pkl")
            return model_data
        else:
            return train_new_model()
    except Exception as e:
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
    st.markdown('<h1 class="main-header">Newborn Health Monitoring and Risk Assessment System</h1>', unsafe_allow_html=True)
    
    # Load model
    model_data = load_model()
    
    if model_data is None:
        st.error("❌ Unable to load or train model. Please check your data files.")
        return
    
    # Main container
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    
    # Form sections
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<h3 class="section-title">Basic Newborn Data</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        gestational_age = st.number_input("Gestational Age (weeks)", min_value=0.0, max_value=50.0, value=38.0, step=0.1)
        birth_length = st.number_input("Birth Length (cm)", min_value=0.0, max_value=100.0, value=50.0, step=0.1)
        gender = st.selectbox("Gender", ["Female", "Male"])
    
    with col2:
        birth_weight = st.number_input("Birth Weight (kg)", min_value=0.0, max_value=10.0, value=3.2, step=0.01)
        birth_head_circumference = st.number_input("Birth Head Circumference (cm)", min_value=0.0, max_value=50.0, value=35.0, step=0.1)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<h3 class="section-title">Current Monitoring Data</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        age_days = st.number_input("Age (days)", min_value=0, max_value=365, value=7)
        current_length = st.number_input("Current Length (cm)", min_value=0.0, max_value=120.0, value=51.0, step=0.1)
    
    with col2:
        current_weight = st.number_input("Current Weight (kg)", min_value=0.0, max_value=15.0, value=3.3, step=0.01)
        current_head_circumference = st.number_input("Current Head Circumference (cm)", min_value=0.0, max_value=60.0, value=36.0, step=0.1)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<h3 class="section-title">Vital Signs</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        temperature = st.number_input("Temperature (°C)", min_value=30.0, max_value=45.0, value=37.0, step=0.1)
        respiratory_rate = st.number_input("Respiratory Rate (breaths/min)", min_value=0, max_value=100, value=40)
    
    with col2:
        heart_rate = st.number_input("Heart Rate (bpm)", min_value=0, max_value=300, value=120)
        oxygen_saturation = st.number_input("Oxygen Saturation (%)", min_value=0, max_value=100, value=98)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<h3 class="section-title">Feeding and Output Data</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        feeding_type = st.selectbox("Feeding Type", ["Breastfeeding", "Formula", "Mixed"])
        urine_output = st.number_input("Urine Output Count", min_value=0, max_value=20, value=6)
    
    with col2:
        feeding_frequency = st.number_input("Feeding Frequency (per day)", min_value=0, max_value=20, value=8)
        stool_count = st.number_input("Stool Count", min_value=0, max_value=20, value=3)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<h3 class="section-title">Assessment Data</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        jaundice_level = st.number_input("Jaundice Level (mg/dl)", min_value=0.0, max_value=50.0, value=8.0, step=0.1)
        immunizations_done = st.selectbox("Immunizations Done", ["Yes", "No"])
    
    with col2:
        apgar_score = st.number_input("Apgar Score", min_value=0.0, max_value=10.0, value=8.0, step=0.1)
        reflexes_normal = st.selectbox("Reflexes Normal", ["Yes", "No"])
    
    st.markdown('</div>', unsafe_allow_html=True)
    
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
    
    # Assessment button
    if st.button("Assess Risk Level", type="primary", use_container_width=True):
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
                    risk_level = "At Risk" if prediction == 1 else "No Risk"
                else:
                    if str(prediction).lower() in ['at risk', 'atrisk', '1']:
                        risk_level = "At Risk"
                    else:
                        risk_level = "No Risk"
                
                # Display result
                if risk_level == "At Risk":
                    st.markdown(f'<div class="risk-high"><h3>{risk_level}</h3><p>Needs NICU admission</p></div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="risk-low"><h3>{risk_level}</h3><p>No NICU admission needed</p></div>', unsafe_allow_html=True)
                
                # Recommendations
                st.subheader("Recommendations")
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
    
    # Close main container
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
