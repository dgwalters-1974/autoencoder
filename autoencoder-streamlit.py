import streamlit as st
import pandas as pd
from model import run_model, plot_anomaly_comparison, plot_results_once, plot_contributions, plot_anomaly_breakdown

# Define available currencies
currency_choices = [
    'AED', 'AUD', 'CAD', 'CHF', 'CNY', 'CZK', 'DKK', 'EUR', 'GBP',
    'HUF', 'JPY', 'NOK', 'NZD', 'PLN', 'RUB', 'SEK', 'SGD', 'THB', 
    'TRY', 'USD'
]

# Sidebar
st.sidebar.header("Settings")
uploaded_file = st.sidebar.file_uploader(
    label="Upload .xlsx or .csv file containing fixings",
    type=["xlsx", "csv"]
)
model_choice = st.sidebar.selectbox(
    "Choose a model for anomaly detection",
    ['Autoencoder', 'IsolationForest', 'KNN', 'LOF', 'OCSVM', 'Mahalanobis']
)
button_calculate = st.sidebar.button("Calculate", key='calculate_button')
chart_choice = st.sidebar.selectbox(
    "Select currency to plot",
    currency_choices,
    key="chart_choice"
)
outright_rates = st.sidebar.toggle("Show outright rates")
selected_currencies = st.sidebar.multiselect(
    "Choose currencies for comparison",
    currency_choices,
    default=currency_choices
)

# Initialize session state for results and chart data
if "results" not in st.session_state:
    st.session_state.results = None
if "selected_currency" not in st.session_state:
    st.session_state.selected_currency = None

# Function to calculate results
def main_calculation():
    return run_model(model_choice, uploaded_file)

# Calculate button logic
if button_calculate:
    if uploaded_file:
        with st.spinner("Processing..."):
            st.session_state.results = main_calculation()
            st.session_state.selected_currency = None  # Reset selected currency after calculation
        st.success("Calculation complete!")
    else:
        st.error("Please upload a dataset before calculating.")

# Main app logic
if st.session_state.results:
    # If the selected currency changes, update the session state but do not recalculate
    if st.session_state.selected_currency != chart_choice:
        st.session_state.selected_currency = chart_choice

    results = st.session_state.results
    currency_results = results[st.session_state.selected_currency]
    anomalies = currency_results[currency_results['Is_Anomaly'] == True]

    # Data choice: outright rates or daily differences
    plot_data = results['data'] if outright_rates else results['data_diff']

    # Header
    st.header(
        f"Outliers in Fixings Data (Method: {model_choice})",
        divider="gray"
    )
    st.subheader(f"Currency: {st.session_state.selected_currency}")
    
    # Charts
    st.pyplot(plot_anomaly_comparison(results, selected_currencies))
    st.pyplot(plot_results_once(plot_data, st.session_state.selected_currency, anomalies))
    st.pyplot(plot_contributions(results, st.session_state.selected_currency))
    st.pyplot(plot_anomaly_breakdown(st.session_state.selected_currency, results))
else:
    st.info("Upload a dataset and click 'Calculate' to begin.")



# import streamlit as st
# import pandas as pd
# from model import run_model, plot_anomaly_comparison, plot_results_once, plot_contributions, plot_anomaly_breakdown


# currency_choice = ['AED',
#  'AUD',
#  'CAD',
#  'CHF',
#  'CNY',
#  'CZK',
#  'DKK',
#  'EUR',
#  'GBP',
#  'HUF',
#  'JPY',
#  'NOK',
#  'NZD',
#  'PLN',
#  'RUB',
#  'SEK',
#  'SGD',
#  'THB',
#  'TRY',
#  'USD']

# #currency_choice = set([x[:3] for x in data.columns])

# # Sidebar
# with st.sidebar:
#     uploaded_file = st.file_uploader(label='.xlsx or .csv file containing fixings...',
#                                      type=["xlsx", "csv"])
    
#     model_choice = st.selectbox('Choose model...', ['Autoencoder', 'IsolationForest', 'KNN', 'LOF', 'OCSVM', 'Mahalanobis'])
#     button_state = st.button("Calculate", key = 'recalc_sheet', use_container_width=True)
#     chart_choice = st.selectbox('Choose plot currency...', currency_choice)

#     on = st.toggle("Outright rates")
#     curr = st.multiselect('Choose currencies...', currency_choice, currency_choice)


# # Run model
# def main_calculation():
#     # if model_choice == 'Autoencoder':
#     #     df = run_model(uploaded_file)
#     # else:
#     #     df = run_model_skl(model_choice, uploaded_file)
#     df = run_model(model_choice, uploaded_file)
#     return df


# #vMain body of app
# if uploaded_file is not None and st.session_state.get("recalc_sheet"):
#     results = main_calculation()
#     anomalies = results[chart_choice][results[chart_choice]['Is_Anomaly'] == True]

    
#     if on:
#         plot_choice = results['data']
#     else:
#         plot_choice = results['data_diff']
        

#     st.header(f'Outliers in fixings data (Method: {model_choice}) with contributions for {chart_choice} indices', divider='gray')
#     st.pyplot(plot_anomaly_comparison(results, curr))
#     st.pyplot(plot_results_once(plot_choice, chart_choice, anomalies))

#     st.pyplot(plot_contributions(results, chart_choice))

#     st.pyplot(plot_anomaly_breakdown(chart_choice, results))

# import streamlit as st
# import pandas as pd
# from model import run_model, plot_anomaly_comparison, plot_results_once, plot_contributions, plot_anomaly_breakdown

# # Define available currencies
# currency_choices = [
#     'AED', 'AUD', 'CAD', 'CHF', 'CNY', 'CZK', 'DKK', 'EUR', 'GBP',
#     'HUF', 'JPY', 'NOK', 'NZD', 'PLN', 'RUB', 'SEK', 'SGD', 'THB', 
#     'TRY', 'USD'
# ]

# # Sidebar
# st.sidebar.header("Settings")
# uploaded_file = st.sidebar.file_uploader(
#     label="Upload .xlsx or .csv file containing fixings",
#     type=["xlsx", "csv"]
# )
# model_choice = st.sidebar.selectbox(
#     "Choose a model for anomaly detection",
#     ['Autoencoder', 'IsolationForest', 'KNN', 'LOF', 'OCSVM', 'Mahalanobis']
# )
# button_calculate = st.sidebar.button("Calculate", key='calculate_button')
# chart_choice = st.sidebar.selectbox(
#     "Select currency to plot",
#     currency_choices
# )
# outright_rates = st.sidebar.toggle("Show outright rates")
# selected_currencies = st.sidebar.multiselect(
#     "Choose currencies for comparison",
#     currency_choices,
#     default=currency_choices
# )

# # Initialize session state for results
# if "results" not in st.session_state:
#     st.session_state.results = None

# # Function to calculate results
# def main_calculation():
#     return run_model(model_choice, uploaded_file)

# # Calculate button logic
# if button_calculate:
#     if uploaded_file:
#         with st.spinner("Processing..."):
#             st.session_state.results = main_calculation()
#         st.success("Calculation complete!")
#     else:
#         st.error("Please upload a dataset before calculating.")

# # Main app logic
# if st.session_state.results:
#     results = st.session_state.results
#     anomalies = results[chart_choice][results[chart_choice]['Is_Anomaly'] == True]

#     # Data choice: outright rates or daily differences
#     plot_data = results['data'] if outright_rates else results['data_diff']

#     # Header
#     st.header(
#         f"Outliers in Fixings Data (Method: {model_choice})",
#         divider="gray"
#     )
#     st.subheader(f"Currency: {chart_choice}")
    
#     # Charts
#     st.pyplot(plot_anomaly_comparison(results, selected_currencies))
#     st.pyplot(plot_results_once(plot_data, chart_choice, anomalies))
#     st.pyplot(plot_contributions(results, chart_choice))
#     st.pyplot(plot_anomaly_breakdown(chart_choice, results))
# else:
#     st.info("Upload a dataset and click 'Calculate' to begin.")

