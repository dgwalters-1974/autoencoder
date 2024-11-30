import streamlit as st
from model import run_model, plot_anomaly_comparison, plot_results_once, plot_contributions, plot_anomaly_breakdown, run_model_skl

currency_choice = ['EUR', 'USD', 'JPY', 'AUD', 'NZD', 'CAD']

# Sidebar
with st.sidebar:
    uploaded_file = st.file_uploader(label='.xlsx or .csv file containing fixings...',
                                     type=["xlsx", "csv"])
    
    model_choice = st.selectbox('Choose model...', ['Autoencoder', 'IsolationForest', 'KNN', 'LOF', 'OCSVM', 'Mahalanobis'])
    button_state = st.button("Calculate", key = 'recalc_sheet', use_container_width=True)
    chart_choice = st.selectbox('Choose plot currency...', currency_choice)

    on = st.toggle("Outright rates")
    curr = st.multiselect('Choose currencies...', currency_choice, currency_choice)


# Run model
def main_calculation():
    if model_choice == 'Autoencoder':
        df = run_model(uploaded_file)
    else:
        df = run_model_skl(model_choice, uploaded_file)
    return df



if uploaded_file is not None and st.session_state.get("recalc_sheet"):
    results = main_calculation()
    anomalies = results[chart_choice][results[chart_choice]['Is_Anomaly'] == True]
    
    # Main body of app
    if on:
        plot_choice = results['data']
    else:
        plot_choice = results['data_diff']
        
    #st.write(results['JPY'].head())
    #st.write(anomalies)
    
    st.header(f'Outliers in fixings data (Method: {model_choice}) with contributions for {chart_choice} indices', divider='gray')
    st.pyplot(plot_anomaly_comparison(results, curr))
    st.pyplot(plot_results_once(plot_choice, chart_choice, anomalies))
    
    st.pyplot(plot_contributions(results, chart_choice))
    
    st.pyplot(plot_anomaly_breakdown(chart_choice, results))


