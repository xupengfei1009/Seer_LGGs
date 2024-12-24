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
    
    # 数值型变量
    input_data.append(state_dict['Age'])
    input_data.append(1 if state_dict['Sex'] == 'Male' else 0)
    input_data.append(state_dict['Size'])
    
    # 分类变量编码
    subtype_encoded = encoders['Subtype'].transform([[state_dict['Subtype']]])
    input_data.extend(subtype_encoded.flatten())
    
    surgery_encoded = encoders['Surgery'].transform([[state_dict['Surgery']]])
    input_data.extend(surgery_encoded.flatten())
    
    treatment_encoded = encoders['AdjuvantTreatment'].transform([[state_dict['AdjuvantTreatment']]])
    input_data.extend(treatment_encoded.flatten())
    
    return np.array(input_data)

def plot_survival_curve(times, survival_probs, probs_12m, probs_36m, probs_60m):
    # 确保数据长度一致
    survival_probs = survival_probs.flatten()  # 确保是一维数组
    
    # 创建基础数据框
    df = pd.DataFrame({
        'Time': times,
        'Survival': survival_probs
    })
    
    # 创建图表
    fig = px.line(df, x='Time', y='Survival', 
                  title='Predicted Survival Probability')
    
    # 添加特定时间点的标记
    fig.add_scatter(
        x=[12, 36, 60],
        y=[probs_12m, probs_36m, probs_60m],
        mode='markers+text',
        text=[f"{v*100:.1f}%" for v in [probs_12m, probs_36m, probs_60m]],
        textposition="top center",
        name="Annual Survival",
        showlegend=False
    )
    
    # 更新布局
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
        yaxis_range=[0, 1],
        template='simple_white',
        plot_bgcolor="white",
        font=dict(size=14)
    )
    
    return fig

def main():
    st.title('Survival Prediction Model for Adult Diffuse Low-grade Glioma')

    # 加载设置和模型
    settings, input_keys = load_setting()
    encoders = create_encoders()
    model = load_model()

    # 侧边栏输入表单
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

    # 处理预测
    if submitted:
        input_data = process_input(st.session_state, encoders)
        
        # 预测生存率
        times = np.arange(61)  # 0-60个月
        survival_probs = model.predict_survival(input_data.reshape(1, -1), t=times)
        probs_12m = model.predict_survival(input_data.reshape(1, -1), t=12)[0]  # 1年
        probs_36m = model.predict_survival(input_data.reshape(1, -1), t=36)[0]  # 3年
        probs_60m = model.predict_survival(input_data.reshape(1, -1), t=60)[0]  # 5年
        
        # 绘制生存曲线
        fig = plot_survival_curve(times, survival_probs, probs_12m, probs_36m, probs_60m)
        st.plotly_chart(fig, use_container_width=True)
        
        # 显示具体的生存率数值
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

# 添加说明
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
