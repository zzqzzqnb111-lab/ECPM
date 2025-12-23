#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Endometrial Cancer 5-Year Survival Prediction Tool - Streamlit Web Application
(DeepSurv Model Only)
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Endometrial Cancer 5-Year Survival Prediction Tool",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 2rem 0;
    }
    .prediction-value {
        font-size: 3rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# DeepSurv model definition
class DeepSurv(nn.Module):
    def __init__(self, input_dim):
        super(DeepSurv, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.out = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = self.out(x)
        return x

# Load models and preprocessing objects
@st.cache_resource
def load_models():
    """Load DeepSurv model and preprocessing objects"""
    try:
        scaler = joblib.load('scaler.pkl')
        label_encoders = joblib.load('label_encoders.pkl')
        feature_cols = joblib.load('feature_cols.pkl')
        
        # DeepSurv related files
        H0_60 = joblib.load('deepsurv_H0_60.pkl')
        calibrator = joblib.load('calibrator_deepsurv.pkl')
        
        # Load DeepSurv model
        model = DeepSurv(input_dim=len(feature_cols))
        model.load_state_dict(torch.load('best_deepsurv.pth', map_location='cpu'))
        model.eval()
        
        return scaler, label_encoders, feature_cols, model, H0_60, calibrator
    except Exception as e:
        st.error(f"Failed to load models: {str(e)}")
        return None, None, None, None, None, None

# Prediction function
def predict_survival(input_data, scaler, label_encoders, feature_cols, model, H0_60, calibrator):
    """Make prediction using DeepSurv model"""
    
    # Prepare input data
    X = pd.DataFrame([input_data], columns=feature_cols)
    
    # Encode categorical variables
    for col in feature_cols:
        if col in label_encoders:
            try:
                X[col] = label_encoders[col].transform([str(input_data[col])])[0]
            except:
                # If value not in training set, use most common value
                X[col] = 0
    
    # Standardize
    X_scaled = scaler.transform(X)
    
    # DeepSurv prediction
    try:
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_scaled)
            risk = model(X_tensor).cpu().numpy()[0, 0]
        
        # Calculate 5-year survival probability: S(60) = exp(-H0_60 * exp(risk))
        surv_prob_raw = np.exp(-H0_60 * np.exp(risk))
        surv_prob_raw = np.clip(surv_prob_raw, 0, 1)
        
        # Calibrate using calibrator
        surv_prob = calibrator.transform([surv_prob_raw])[0]
        surv_prob = np.clip(surv_prob, 0, 1)
        
        return surv_prob
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
        return None

# Main interface
def main():
    # Title
    st.markdown('<div class="main-header">🏥 Endometrial Cancer 5-Year Survival Prediction Tool</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Machine Learning-Based Prognostic Risk Assessment System</div>', unsafe_allow_html=True)
    
    # Load models
    scaler, label_encoders, feature_cols, model, H0_60, calibrator = load_models()
    
    if scaler is None:
        st.stop()
    
    # Sidebar instructions
    with st.sidebar:
        st.markdown("### 📋 Instructions")
        st.markdown("""
        **Functionality:**  
        This tool uses the DeepSurv deep learning model to predict 5-year survival probability for endometrial cancer patients.
        
        **How to use:**  
        1. Enter patient's basic information and clinical features on the right
        2. Click the "Calculate Prediction" button
        3. View the DeepSurv model's prediction results
        
        **Notes:**  
        - All fields are required
        - Prediction results are for reference only and cannot replace clinical judgment
        - Please combine with clinical practice and comprehensive assessment
        """)
        
        st.markdown("---")
        st.markdown("### 📊 Model Information")
        st.markdown("""
        - **DeepSurv**: Deep learning survival analysis model
        - Uses neural networks for survival prediction
        - Based on Cox proportional hazards framework
        """)
    
    # Input form
    st.markdown("### 📝 Patient Information Input")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age (years)", min_value=0, max_value=120, value=60, step=1)
        
        race_options = ['White', 'Black', 'Others']
        race = st.selectbox("Race", race_options)
        
        marital_options = ['Married', 'Single/unmarried', 'Divorced/separated', 'Widowed']
        marital_status = st.selectbox("Marital status", marital_options)
        
        histo_options = ['Endometroid', 'Serous', 'Mixed', 'Clear cell', 'Others']
        histo_type = st.selectbox("Histological type", histo_options)
        
        grade_options = ['G1', 'G2', 'G3']
        grade = st.selectbox("Grade", grade_options)
        
        tumor_size_options = ['<5cm', '5-10cm', '≥10cm']
        tumor_size = st.selectbox("Tumor size", tumor_size_options)
        
        lymph_node_options = ['no', 'yes']
        lymph_node = st.selectbox("Lymph node metastasis", lymph_node_options)
        
        metastasis_options = ['M0', 'M1']
        metastasis = st.selectbox("Metastasis", metastasis_options)
    
    with col2:
        tumor_stage_options = ['Localized', 'Regional', 'Distant']
        tumor_stage = st.selectbox("Tumor stage", tumor_stage_options)
        
        figo_options = ['IA', 'IB', 'IC', 'IIA', 'IIC', 'IIIA', 'IIIC', 'IV']
        figo_stage = st.selectbox("FIGO stage", figo_options)
        
        surgery_options = ['no', 'yes']
        surgery = st.selectbox("Surgery", surgery_options)
        
        radio_options = ['no', 'yes']
        radiotherapy = st.selectbox("Radiotherapy", radio_options)
        
        chemo_options = ['no', 'yes']
        chemotherapy = st.selectbox("Chemotherapy", chemo_options)
        
        lymphad_options = ['no', 'yes']
        lymphadenectomy = st.selectbox("Lymphadenectomy", lymphad_options)
        
        income_options = ['<6w', '6-8w', '8-10w', '≥10w']
        income = st.selectbox("Median household income", income_options)
    
    # Prediction button
    st.markdown("---")
    if st.button("🔮 Calculate Prediction", type="primary", use_container_width=True):
        # Prepare input data
        input_data = {
            'Age': age,
            'Race': race,
            'Marital status': marital_status,
            'Histological type': histo_type,
            'Grade': grade,
            'Tumor size': tumor_size,
            'Lymph node metastasis': lymph_node,
            'Metastasis': metastasis,
            'Tumor stage': tumor_stage,
            'FIGO stage': figo_stage,
            'Surgery': surgery,
            'Radiotherapy': radiotherapy,
            'Chemotherapy': chemotherapy,
            'Lymphadenectomy': lymphadenectomy,
            'Median household income': income
        }
        
        # Prediction
        with st.spinner("Calculating prediction using DeepSurv model..."):
            survival_prob = predict_survival(input_data, scaler, label_encoders, feature_cols, model, H0_60, calibrator)
        
        # Display results
        st.markdown("---")
        st.markdown("### 📊 Prediction Results")
        
        if survival_prob is not None:
            # Main prediction result box
            st.markdown(f"""
            <div class="prediction-box">
                <h3>5-Year Survival Probability</h3>
                <div class="prediction-value">{survival_prob*100:.1f}%</div>
                <p>Predicted by DeepSurv Deep Learning Model</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Risk level assessment
            st.markdown("---")
            st.markdown("#### Risk Level Assessment")
            
            if survival_prob >= 0.8:
                risk_level = "Low Risk"
                risk_color = "🟢"
                risk_desc = "Good prognosis with higher 5-year survival probability"
            elif survival_prob >= 0.6:
                risk_level = "Low-Moderate Risk"
                risk_color = "🟡"
                risk_desc = "Moderate prognosis, close follow-up recommended"
            elif survival_prob >= 0.4:
                risk_level = "Moderate-High Risk"
                risk_color = "🟠"
                risk_desc = "Prognosis requires attention, active treatment and monitoring recommended"
            else:
                risk_level = "High Risk"
                risk_color = "🔴"
                risk_desc = "Poor prognosis, intensive treatment and care recommended"
            
            st.info(f"{risk_color} **{risk_level}**: {risk_desc}")
            
            # Disclaimer
            st.markdown("---")
            st.markdown("""
            <div class="info-box">
                <strong>⚠️ Important Notice:</strong><br>
                This prediction tool is based on the DeepSurv deep learning model. Prediction results are for reference only and cannot replace professional clinical judgment.
                Actual prognosis is influenced by multiple factors. Please combine with patient-specific conditions, clinical experience, and the latest research evidence for comprehensive assessment.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.error("DeepSurv model prediction failed. Please check input data or contact technical support.")

if __name__ == "__main__":
    main()
