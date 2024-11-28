import streamlit as st
from model import run_model, plot_anomaly_comparison, plot_results_once

currency_choice = ['EUR', 'USD', 'JPY', 'AUD', 'NZD', 'CAD']

# Sidebar
with st.sidebar:
    uploaded_file = st.file_uploader(label='.xlsx or .csv file containing fixings...',
                                     type=["xlsx", "csv"])

with st.sidebar:    
    curr = st.multiselect('Choose currencies...', currency_choice, currency_choice)
    
with st.sidebar:
    chart_choice = st.selectbox('Choose plot currency...', currency_choice)

with st.sidebar:
    on = st.toggle("Outright rates")


# Do this only when initially loaded  
@st.cache_data
def main_calculation():
    df = run_model(uploaded_file)
    return df



if uploaded_file is not None:
    results = main_calculation()
    
    # Main body of app
    if on:
        plot_choice = results['data']
    else:
        plot_choice = results['data_diff']
    
    st.header('Autoencoder')
    st.pyplot(plot_anomaly_comparison(results, curr))
    st.pyplot(plot_results_once(plot_choice, chart_choice, results[chart_choice]))

