import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import pickle
from sklearn.preprocessing import OneHotEncoder

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
    
    subtype_encoded = encoders['Subtype'].transform([[state_dict['Subtype']]])
    input_data.extend(subtype_encoded.flatten())
    
    surgery_encoded = encoders['Surgery'].transform([[state_dict['Surgery']]])
    input_data.extend(surgery_encoded.flatten())
    
    treatment_encoded = encoders['AdjuvantTreatment'].transform([[state_dict['AdjuvantTreatment']]])
    input_data.extend(treatment_encoded.flatten())
    
    return np.array(input_data)

def plot_survival_curve(times, survival_probs, probs_12m, probs_36m, probs_60m):
    df = pd.DataFrame({
        'Time': times,
        'Survival': survival_probs
    })
    
    fig = px.line(df, x='Time', y='Survival')
    
    fig.update_traces(
        line=dict(color='#2E75B6', width=3),
        name='Survival Probability'
    )
    
    marker_times = [12, 36, 60]
    marker_probs = [probs_12m, probs_36m, probs_60m]
    
    text_positions = []
    for prob in marker_probs:
        if prob > 0.9:
            text_positions.append('bottom center')
        elif prob < 0.1:
            text_positions.append('top center')
        else:
            text_positions.append('middle right')
    
    fig.add_trace(
        go.Scatter(
            x=marker_times,
            y=marker_probs,
            mode='markers+text',
            marker=dict(
                color='#E74C3C',
                size=12,
                symbol='circle'
            ),
            text=[f"{v*100:.1f}%" for v in marker_probs],
            textposition=text_positions,
            textfont=dict(
                size=14,
                color='#2C3E50'
            ),
            name='Annual Survival',
            showlegend=False
        )
    )
    
    for t in marker_times:
        fig.add_shape(
            type='line',
            x0=t, x1=t,
            y0=0, y1=1,
            line=dict(
                color='#95A5A6',
                width=1,
                dash='dash'
            )
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
        xaxis=dict(
            title="Time (Months)",
            gridcolor='#EAECEE',
            zeroline=True,
            zerolinecolor='#95A5A6',
            zerolinewidth=1,
            range=[0, 65],  
            tickmode='array',
            tickvals=[0, 12, 24, 36, 48, 60],
            ticktext=['0', '1y', '2y', '3y', '4y', '5y']
        ),
        yaxis=dict(
            title="Survival Probability",
            gridcolor='#EAECEE',
            zeroline=True,
            zerolinecolor='#95A5A6',
            zerolinewidth=1,
            range=[0, 1],
            tickformat='.0%'
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(
            family="Arial",
            size=14,
            color="#2C3E50"
        ),
        margin=dict(t=100, l=80, r=80, b=80),
        showlegend=False,
        hovermode='x unified'
    )
    
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#EAECEE')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#EAECEE')
    
    return fig

def main():
    st.title('Survival Prediction Model for Adult Diffuse Low-grade Glioma')

    settings, input_keys = load_setting()
    encoders = create_encoders()
    model = load_model()

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
            
            submitted = st.form_submit_button("Predict")

    if submitted:
        input_data = process_input(st.session_state, encoders)
        input_data_reshaped = input_data.reshape(1, -1)
    
        times = np.arange(61)  
        survival_probs = []
    
        for t in times:
            prob = model.predict_survival(input_data_reshaped, t=t)[0]
            survival_probs.append(prob)
    
        survival_probs = np.array(survival_probs)
    
        probs_12m = model.predict_survival(input_data_reshaped, t=12)[0]
        probs_36m = model.predict_survival(input_data_reshaped, t=36)[0]
        probs_60m = model.predict_survival(input_data_reshaped, t=60)[0]
    
    fig = plot_survival_curve(times, survival_probs, probs_12m, probs_36m, probs_60m)
    st.plotly_chart(fig, use_container_width=True, height=600)
    
    st.markdown("### Prediction Results")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("1-year Survival", f"{probs_12m*100:.1f}%")
    with col2:
        st.metric("3-year Survival", f"{probs_36m*100:.1f}%")
    with col3:
        st.metric("5-year Survival", f"{probs_60m*100:.1f}%")

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
