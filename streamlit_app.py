import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import pickle
from sklearn.preprocessing import OneHotEncoder
from pysurvival.utils import load_model

st.set_page_config(layout="wide")

@st.cache_data(show_spinner=False)
def load_setting():
    settings = {
        'Age': {'values': [19, 85], 'type': 'slider', 'init_value': 30},
        'Sex': {'values': ["Female", "Male"], 'type': 'selectbox', 'init_value': 0},
        'Size': {'values': [1, 100], 'type': 'slider', 'init_value': 50},
        'Subtype': {
            'values': ["AST(IDH-mutant)", "AST(IDH-wild)", "OLI(IDH-mutant)"],
            'type': 'selectbox',
            'init_value': 0
        },
        'Surgery': {
            'values': ["Biopsy", "GTR", "PR", "STR"],
            'type': 'selectbox',
            'init_value': 0
        },
        'AdjuvantTreatment': {
            'values': ["CRT", "CT", "None", "RT"],
            'type': 'selectbox',
            'init_value': 0
        }
    }
    
    input_keys = ['Age', 'Sex', 'Size', 'Subtype', 'Surgery', 'AdjuvantTreatment']
    return settings, input_keys

@st.cache_resource
def create_encoders():
    encoders = {
        'Subtype': OneHotEncoder(sparse=False, drop=None),
        'Surgery': OneHotEncoder(sparse=False, drop=None),
        'AdjuvantTreatment': OneHotEncoder(sparse=False, drop=None)
    }
    encoders['Subtype'].fit([["AST(IDH-mutant)"], ["AST(IDH-wild)"], ["OLI(IDH-mutant)"]])
    encoders['Surgery'].fit([["Biopsy"], ["GTR"], ["PR"], ["STR"]])
    encoders['AdjuvantTreatment'].fit([["CRT"], ["CT"], ["None"], ["RT"]])
    
    return encoders

@st.cache_resource
def load_model():
    with open('DCPHModelFinal.pkl', 'rb') as f:
        model = pickle.load(f)
    return model


def process_input(state_dict, encoders):
    input_data = []
    input_data.append(state_dict['Age'])
    input_data.append(1 if state_dict['Sex'] == 'Male' else 0)
    input_data.append(state_dict['Size'])
    
    for feature in ['Subtype', 'Surgery', 'AdjuvantTreatment']:
        encoded = encoders[feature].transform([[state_dict[feature]]])
        input_data.extend(encoded.flatten())
    
    return np.array(input_data)

def predict(input_data, model):

    times = np.arange(61)  
    survival_probs = model.predict_survival(input_data.reshape(1, -1), times)
    
    probs_12m = model.predict_survival(input_data.reshape(1, -1), np.array([12]))[0]
    probs_36m = model.predict_survival(input_data.reshape(1, -1), np.array([36]))[0]
    probs_60m = model.predict_survival(input_data.reshape(1, -1), np.array([60]))[0]
    
    return survival_probs, probs_12m, probs_36m, probs_60m

st.title('Survival Prediction Model for Adult Diffuse Low-grade Glioma')

def plot_survival_curve(times, survival_probs, probs_12m, probs_36m, probs_60m):
    df = pd.DataFrame({
        'Time': times,
        'Survival': survival_probs.flatten()
    })
    
    fig = px.line(df, x='Time', y='Survival',
                  labels={'Time': 'Time (months)',
                         'Survival': 'Survival Probability'},
                  title='Predicted Survival Curve')
    
    fig.add_scatter(
        x=[12, 36, 60],
        y=[probs_12m, probs_36m, probs_60m],
        mode='markers+text',
        text=[f'{p*100:.1f}%' for p in [probs_12m, probs_36m, probs_60m]],
        textposition="top center",
        name='Annual Survival',
        showlegend=False
    )
    
    fig.update_layout(
        xaxis_range=[0, 60],
        yaxis_range=[0, 1],
        template='simple_white',
        plot_bgcolor='white'
    )
    
    return fig

def main():
    
    settings, input_keys = load_setting()
    encoders = create_encoders()
    model = load_model()
    
    with st.sidebar:
        st.header("Patient Information")
        with st.form("patient_form"):
            for key in input_keys:
                setting = settings[key]
                if setting['type'] == 'slider':
                    st.session_state[key] = st.slider(
                        f"{key}",
                        min_value=setting['values'][0],
                        max_value=setting['values'][1],
                        value=setting['init_value']
                    )
                else:  # selectbox
                    st.session_state[key] = st.selectbox(
                        f"{key}",
                        options=setting['values'],
                        index=setting['init_value']
                    )
            
            submitted = st.form_submit_button("Predict")
    
    if submitted:
        input_data = process_input(st.session_state, encoders)
    
        survival_probs, probs_12m, probs_36m, probs_60m = predict(input_data, model)
        
        st.subheader("Prediction Results")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("1-year Survival", f"{probs_12m*100:.1f}%")
        with col2:
            st.metric("3-year Survival", f"{probs_36m*100:.1f}%")
        with col3:
            st.metric("5-year Survival", f"{probs_60m*100:.1f}%")
        
        fig = plot_survival_curve(
            np.arange(61),
            survival_probs,
            probs_12m,
            probs_36m,
            probs_60m
        )
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()

st.markdown("---")
st.markdown("### User Guide")
st.markdown("""
1. Enter patient information in the left panel
2. Click the "Predict" button to generate prediction results
3. View the survival curve and specific predicted values
""")

st.markdown("### Notes")
st.markdown("""
- This model is for research reference only and should not be used as the sole basis for clinical decisions
- Prediction results should be evaluated comprehensively in conjunction with the patient's specific conditions
""")
