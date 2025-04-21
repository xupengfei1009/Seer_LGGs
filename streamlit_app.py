import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import pickle
from pysurvival.utils import load_model

st.set_page_config(
    page_title="LGG Survival Prediction",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #00A5E0;
        color: white;
        border-radius: 5px;
        height: 3em;
        font-weight: bold;
    }
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
        color: #00A5E0;
    }
    .title-text {
        font-size: 2.5rem;
        color: #1E3D59;
        text-align: center;
        margin-bottom: 2rem;
        padding: 1rem;
        border-bottom: 2px solid #00A5E0;
    }
    .subtitle-text {
        font-size: 1.5rem;
        color: #1E3D59;
        margin-bottom: 1rem;
    }
    .info-box {
        padding: 1rem;
        border-radius: 5px;
        background-color: #f8f9fa;
        border-left: 5px solid #00A5E0;
        margin-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def load_setting():
    settings = {
        'Age': {'values': [19, 82], 'type': 'slider', 'init_value': 30, 'add_after': ' years'},
        'Sex': {'values': ["Female", "Male"], 'type': 'selectbox', 'init_value': 0, 'add_after': ''},
        'Size': {'values': [4, 110], 'type': 'slider', 'init_value': 50, 'add_after': ' mm'},
        'Subtype': {
            'values': ["OLI(IDH-mutant)", "AST(IDH-mutant)"],
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
def load_model():
    with open('DCPHModelFinal.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

def process_input(state_dict):
    input_data = []
    
    input_data.append(state_dict['Age'])
    input_data.append(1 if state_dict['Sex'] == 'Male' else 0)
    input_data.append(state_dict['Size'])
    input_data.append(1 if state_dict['Subtype'] == 'AST(IDH-mutant)' else 0)
    
    surgery_type = state_dict['Surgery']
    input_data.extend([
        1 if surgery_type == 'GTR' else 0,
        1 if surgery_type == 'PR' else 0,
        1 if surgery_type == 'STR' else 0
    ])
    
    treatment_type = state_dict['AdjuvantTreatment']
    input_data.extend([
        1 if treatment_type == 'CRT' else 0,
        1 if treatment_type == 'CT' else 0,
        1 if treatment_type == 'None' else 0,
        1 if treatment_type == 'RT' else 0
    ])
    
    return np.array(input_data)

def predict():
    input_data = process_input(st.session_state)
    survival = model.predict_survival(input_data.reshape(1, -1), t=None)
    
    
    time_points = survival.shape[1]
    
    data = {
        'survival': survival.flatten()[:time_points],
        'times': list(range(time_points)),
        'No': len(st.session_state['patients']) + 1,
        'arg': {key: st.session_state[key] for key in input_keys},
        
        '1-year': survival[0, min(11, time_points-1)],  
        '3-year': survival[0, min(35, time_points-1)],  
        '5-year': survival[0, min(59, time_points-1)]   
    }
    
    st.session_state['patients'].append(data)

def plot_survival():
    pd_data = pd.concat([
        pd.DataFrame({
            'Survival': item['survival'],
            'Time': item['times'],
            'Patient': [f"Patient {item['No']}" for _ in item['times']]
        }) for item in st.session_state['patients']
    ])
    
    fig = px.line(pd_data, x="Time", y="Survival", color='Patient',
                  range_x=[0, pd_data['Time'].max()], range_y=[0, 1])  
    
    last_patient = st.session_state['patients'][-1]
    time_points = len(last_patient['times'])
    
    plot_points = []
    if time_points >= 12:  
        plot_points.append((12, last_patient['1-year']))
    if time_points >= 36:  
        plot_points.append((36, last_patient['3-year']))
    if time_points >= 60:  
        plot_points.append((60, last_patient['5-year']))
    
    if plot_points:
        x_values, y_values = zip(*plot_points)
        fig.add_scatter(
            x=list(x_values),
            y=list(y_values),
            mode='markers+text',
            text=[f"{v*100:.1f}%" for v in y_values],
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

st.markdown('<p class="title-text">🧠 Survival Prediction Model for Adult Diffuse Low-grade Glioma</p>', unsafe_allow_html=True)

settings, input_keys = load_setting()
model = load_model()

if 'patients' not in st.session_state:
    st.session_state['patients'] = []

with st.sidebar:
    st.markdown('<p class="subtitle-text">Patient Information</p>', unsafe_allow_html=True)
    with st.form("prediction_form"):
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        
        st.slider("Age (year)", 
                 settings['Age']['values'][0],
                 settings['Age']['values'][1],
                 settings['Age']['init_value'],
                 key='Age',
                 help="Patient's age in years")
        
        st.selectbox("Sex",
                    settings['Sex']['values'],
                    index=settings['Sex']['init_value'],
                    key='Sex')
        
        st.slider("Tumor Size (mm)",
                 settings['Size']['values'][0],
                 settings['Size']['values'][1],
                 settings['Size']['init_value'],
                 key='Size',
                 help="Maximum tumor diameter in millimeters")
        
        st.selectbox("Subtype",
                    settings['Subtype']['values'],
                    index=settings['Subtype']['init_value'],
                    key='Subtype',
                    help="Tumor molecular subtype")
        
        st.selectbox("Extent of Resection",
                    settings['Surgery']['values'],
                    index=settings['Surgery']['init_value'],
                    key='Surgery',
                    help="GTR: Gross Total Resection\nSTR: Subtotal Resection\nPR: Partial Resection")
        
        st.selectbox("Adjuvant Treatment",
                    settings['AdjuvantTreatment']['values'],
                    index=settings['AdjuvantTreatment']['init_value'],
                    key='AdjuvantTreatment',
                    help="CRT: Chemoradiotherapy\nCT: Chemotherapy\nRT: Radiotherapy")
        
        st.markdown('</div>', unsafe_allow_html=True)
        submitted = st.form_submit_button("Generate Prediction", on_click=predict)


if st.session_state['patients']:
    with st.container():
        st.markdown('<p class="subtitle-text">Survival Probability Curve</p>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([6, 1]) 
        with col2:
            if st.button("Reset", key="reset_button", help="Clear all survival curves"):
                st.session_state['patients'] = []
                st.rerun()
        
        plot_survival()
        
        st.markdown('<p class="subtitle-text">Prediction Results</p>', unsafe_allow_html=True)
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


st.markdown("---")
st.markdown('<p class="subtitle-text">📋 User Guide</p>', unsafe_allow_html=True)
with st.expander("Click to expand", expanded=True):
    st.markdown("""
    <div class="info-box">
    1. Enter patient information in the left panel
    2. Click the "Generate Prediction" button
    3. View the survival curve and predicted survival rates
    </div>
    """, unsafe_allow_html=True)

st.markdown('<p class="subtitle-text">⚠️ Important Notes</p>', unsafe_allow_html=True)
with st.expander("Click to expand", expanded=True):
    st.markdown("""
    <div class="info-box">
    - This model is for research reference only
    - Predictions should not be used as the sole basis for clinical decisions
    - Always consider patient-specific factors and clinical expertise
    - Consult with healthcare professionals for medical decisions
    </div>
    """, unsafe_allow_html=True)