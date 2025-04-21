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
        'Age': {'values': [19, 82], 'type': 'slider', 'init_value': 30, 'add_after': ' years'},
        'Sex': {'values': ["Female", "Male"], 'type': 'selectbox', 'init_value': 0, 'add_after': ''},
        'Size': {'values': [4, 110], 'type': 'slider', 'init_value': 50, 'add_after': ' mm'},
        'Subtype': {
            'values': ["OLI(IDH-mutant)","AST(IDH-mutant)"],
            'type': 'selectbox',
            'init_value': 0,
            'add_after': ''
        },
        'Surgery': {
            'values': ["GTR", "PR", "STR"],
            'type': 'selectbox',
            'init_value': 0,
            'add_after': ''
        },
        'AdjuvantTreatment': {
            'values': ["CRT", "CT", "None", "RT"],
            'type': 'selectbox',
            'init_value': 0,
            'add_after': ''
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
    encoders['Subtype'].fit([["AST(IDH-mutant)"], ["OLI(IDH-mutant)"]])
    encoders['Surgery'].fit([["GTR"], ["PR"], ["STR"]])
    encoders['AdjuvantTreatment'].fit([["CRT"], ["CT"], ["None"], ["RT"]])
    
    return encoders

@st.cache_resource
def load_model():
    with open('your_model_path.pkl', 'rb') as f:
        model = pickle.load(f)
    return model


def process_input(state_dict):
    input_data = []
    
    input_data.append(state_dict['Age'])
    
    input_data.append(1 if state_dict['Sex'] == 'Male' else 0)
    
    input_data.append(state_dict['Size'])
    
    subtype_encoded = encoders['Subtype'].transform([[state_dict['Subtype']]])
    input_data.extend(subtype_encoded.flatten())
    
    surgery_encoded = encoders['Surgery'].transform([[state_dict['Surgery']]])
    input_data.extend(surgery_encoded.flatten())
    
    treatment_encoded = encoders['AdjuvantTreatment'].transform([[state_dict['AdjuvantTreatment']]])
    input_data.extend(treatment_encoded.flatten())
    
    return np.array(input_data)

def predict():
    input_data = process_input(st.session_state)
    
    survival = model.predict_survival(input_data.reshape(1, -1), t=None)
    
    data = {
        'survival': survival.flatten()[:60],
        'times': list(range(60)),
        'No': len(st.session_state['patients']) + 1,
        'arg': {key: st.session_state[key] for key in input_keys},
        '1-year': survival[0, 11],
        '3-year': survival[0, 35],
        '5-year': survival[0, 59]
    }
    
    st.session_state['patients'].append(data)
    print('Prediction done')

st.title('Survival Prediction Model for Adult Diffuse Low-grade Glioma')

def plot_survival():
    pd_data = pd.concat([
        pd.DataFrame({
            'Survival': item['survival'],
            'Time': item['times'],
            'Patient': [f"Patient {item['No']}" for _ in item['times']]
        }) for item in st.session_state['patients']
    ])
    
    fig = px.line(pd_data, x="Time", y="Survival", color='Patient',
                  range_x=[0, 60], range_y=[0, 1])
    
    last_patient = st.session_state['patients'][-1]
    fig.add_scatter(
        x=[12, 36, 60], 
        y=[last_patient['1-year'], last_patient['3-year'], last_patient['5-year']],
        mode='markers+text',
        text=[f"{v*100:.1f}%" for v in [last_patient['1-year'], last_patient['3-year'], last_patient['5-year']]],
        textposition="top center",
        name="Annual Survival",
        showlegend=False
    )
    
    fig.update_layout(
        title={
            'text': 'Predicted Survival Probability',
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=24)
        },
        xaxis_title="Time (Months)",
        yaxis_title="Survival Probability",
        template='simple_white',
        plot_bgcolor="white",
        font=dict(size=14)
    )
    
    st.plotly_chart(fig, use_container_width=True)

settings, input_keys = load_setting()
encoders = create_encoders()
model = load_model()

if 'patients' not in st.session_state:
    st.session_state['patients'] = []


with st.sidebar:
    with st.form("prediction_form"):
        
        st.slider("Age (year)", 
                 settings['Age']['values'][0],
                 settings['Age']['values'][1],
                 settings['Age']['init_value'],
                 key='Age')
        
        st.selectbox("Sex",
                    settings['Sex']['values'],
                    index=settings['Sex']['init_value'],
                    key='Sex')
        
        st.slider("Tumor Size (mm)",
                 settings['Size']['values'][0],
                 settings['Size']['values'][1],
                 settings['Size']['init_value'],
                 key='Size')
        
        st.selectbox("Subtype",
                    settings['Subtype']['values'],
                    index=settings['Subtype']['init_value'],
                    key='Subtype')
        
        st.selectbox("Extent of Resection",
                    settings['Surgery']['values'],
                    index=settings['Surgery']['init_value'],
                    key='Surgery')
        
        st.selectbox("Adjuvant Treatment",
                    settings['AdjuvantTreatment']['values'],
                    index=settings['AdjuvantTreatment']['init_value'],
                    key='AdjuvantTreatment')
        
        submitted = st.form_submit_button("Predict", on_click=predict)

if st.session_state['patients']:
    plot_survival()
    
    st.markdown("Prediction Results")
    col1, col2, col3 = st.columns(3)
    last_patient = st.session_state['patients'][-1]
    
    with col1:
        st.metric(
            label="1-year Survival Rate",
            value=f"{last_patient['1-year']*100:.1f}%"
        )
    with col2:
        st.metric(
            label="3-year Survival Rate",
            value=f"{last_patient['3-year']*100:.1f}%"
        )
    with col3:
        st.metric(
            label="5-year Survival Rate",
            value=f"{last_patient['5-year']*100:.1f}%"
        )


# 添加说明信息
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
