import streamlit as st
import pandas as pd
import plotly.express as px
from typing import Tuple
from catboost import CatBoostClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np

st.set_page_config(
    page_title="Inspection Data Analysis",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_data(file_path: str) -> pd.DataFrame:
    data = pd.read_csv(file_path)
    df = data.drop(columns=[
        'Customer', 'GRSNo', 'MFG_PhysicalInspection', 
        'Quantity', 'SampleQTY', 'Status', 'InspectorTime'
    ])
    df.dropna(subset=['InspectionResult'], inplace=True)
    return df

def group_data(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby('MPN_PhysicalInspection')['InspectionResult'].value_counts().unstack().fillna(0)
    grouped['PassRate'] = grouped['Pass'] / (grouped['Pass'] + grouped['Reject'])
    return grouped

def calculate_statistics(df: pd.DataFrame, request_value: str, column: str) -> Tuple[int, int, int, float]:
    filtered_df = df[df[column] == request_value]
    pass_count = filtered_df['InspectionResult'].value_counts().get('Pass', 0)
    reject_count = filtered_df['InspectionResult'].value_counts().get('Reject', 0)
    total_count = len(filtered_df)
    pass_rate = pass_count / total_count if total_count > 0 else 0
    return pass_count, reject_count, total_count, pass_rate

def find_mfg_saprequest(df: pd.DataFrame, mpn_physicalinspection: str) -> str:
    filtered_df = df[df['MPN_PhysicalInspection'] == mpn_physicalinspection]
    if not filtered_df.empty:
        return filtered_df['MFG_SAPRequest'].iloc[0]
    return None

def plot_statistics(request_value: str, pass_count: int, reject_count: int, total_count: int, pass_rate: float, title: str):
    labels = ['Pass', 'Reject']
    counts = [pass_count, reject_count]

    fig = px.bar(
        x=labels,
        y=counts,
        labels={'x': 'Result', 'y': 'Count'},
        title=f'Inspection Results for {title}',
        color=labels,
        color_discrete_map={'Pass': 'lightgreen', 'Reject': 'darkred'},
        text=counts  
    )

    fig.update_layout(
        yaxis=dict(
            title='Count',
            titlefont_size=16,
            tickfont_size=14,
            tickformat='.1f',
            title_standoff=25  
        ),
        xaxis=dict(
            title='Result',
            titlefont_size=16,
            tickfont_size=14
        ),
        title=dict(
            text=f'Inspection Results for {title}',
            x=0.5,
            xanchor='center'
        ),
        uniformtext_minsize=8,
        uniformtext_mode='hide',
        margin=dict(t=100)  
    )

    fig.update_traces(
        texttemplate='<b>%{text}</b>',  # Make text bold
        textposition='outside'  # Position text outside the bars
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown(f'**<span style="font-size: 20px;">Pass Rate for {title}: {pass_rate:.2%}</span>**', unsafe_allow_html=True)
    st.markdown(f'**<span style="font-size: 18px;">Number of Passes: {pass_count}</span>**', unsafe_allow_html=True)
    st.markdown(f'**<span style="font-size: 18px;">Number of Rejections: {reject_count}</span>**', unsafe_allow_html=True)
    st.markdown(f'**<span style="font-size: 18px;">Total Number of Inspections: {total_count}</span>**', unsafe_allow_html=True)

def predict_inspection_result(model, MFG_SAPRequest: str, MPN_PhysicalInspection: str) -> str:
    input_data = np.array([[MFG_SAPRequest, MPN_PhysicalInspection]])
    prediction = model.predict(input_data)
    return prediction[0] 

def main():
    st.title("Inspection Result Data Analysis")
    #st.write("Analyze inspection data to determine pass and reject rates for different requests.")
    
    model_path = r"models/final_model.cbm"
    loaded_model = CatBoostClassifier()
    loaded_model.load_model(model_path)

    file_path = r"C:\Users\3904650\Desktop\ML\PassAndReject\GTime.csv"
    df = load_data(file_path)
    grouped = group_data(df)

    with st.sidebar:
        st.header("Enter MPN_PhysicalInspection value:")
        request_value_mpn = st.text_input(" ", 'B78555A2448A  4') # MPN_PhysicalInspection
        if not request_value_mpn:
            st.warning("Please enter an MPN_PhysicalInspection value.")
            return       
    mfg_saprequest = find_mfg_saprequest(df, request_value_mpn) # MFG_SAPRequest
    
    if mfg_saprequest and request_value_mpn:
        predicted_result = predict_inspection_result(loaded_model, mfg_saprequest, request_value_mpn)
        predicted_result_label = 'Pass' if predicted_result == 0 else 'Reject'
        st.markdown(f'**<span style="font-size: 20px;">Predicted Inspection Result for {request_value_mpn}: {predicted_result_label}</span>**', unsafe_allow_html=True)
    else:
        st.markdown(f'**<span style="font-size: 20px; color: red;">No corresponding MFG_SAPRequest or MPN_PhysicalInspection found for {request_value_mpn}</span>**', unsafe_allow_html=True)
    
    
    pass_count, reject_count, total_count, pass_rate = calculate_statistics(df, request_value_mpn, 'MPN_PhysicalInspection')
    plot_statistics(request_value_mpn, pass_count, reject_count, total_count, pass_rate, f'MPN_PhysicalInspection {request_value_mpn}')

    # mfg_saprequest = find_mfg_saprequest(df, request_value_mpn)
    if mfg_saprequest:
        st.markdown(
            f"""
            <div style="background-color: #d1e7dd; padding: 10px; border-radius: 10px; text-align: center; margin-top: 10px;">
                <h2 style="color: #0f5132; font-weight: bold;">Corresponding MFG_SAPRequest: {mfg_saprequest}</h2>
                <p style="font-size: 16px; color: #0f5132;">Please scroll down to view more information.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.write('\n')
        st.write('\n')
        st.write('\n')
        pass_count, reject_count, total_count, pass_rate = calculate_statistics(df, mfg_saprequest, 'MFG_SAPRequest')
        plot_statistics(mfg_saprequest, pass_count, reject_count, total_count, pass_rate, f'MFG_SAPRequest {mfg_saprequest}')
    else:
        st.markdown(
            """
            <div style="background-color: #ffcccb; padding: 10px; border-radius: 5px; text-align: center;">
                <h2 style="color: red;">No corresponding MFG_SAPRequest found.</h2>
                <p style="font-size: 16px; color: black;">Please scroll down to view more information.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

if __name__ == "__main__":
    main()
