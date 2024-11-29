import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import mahalanobis


"""
This file contains the model code along with some plotting functions

"""


def run_model(file_path = 'C:/Users/dgwal/autoencoder-sf/autoencoder/fixings_SmartXM_no empty_max indices.csv'):
    """
    Function to load data from .csv file and run 
    autoencoder for selection of currencies. Outputs 
    a dictionary of results.
    """
    torch.manual_seed(0)

    results_overall = {}

    # Load the dataset
    #file_path = 'C:/Users/dgwal/autoencoder-sf/autoencoder/fixings_SmartXM_no empty_max indices.csv'
    data = pd.read_csv(file_path)

    # Preprocessing: Convert 'Date' to datetime and set it as index
    data['Date'] = pd.to_datetime(data['Date'], format='%d-%b-%y')
    data.set_index('Date', inplace=True)

    # Compute daily differences
    data_daily_diff = data.diff().dropna()

    results_overall['data'] = data
    results_overall['data_diff'] = data_daily_diff
    
    
    currencies = ['JPY', 'EUR', 'USD', 'AUD', 'NZD', 'CAD', 'GBP']

    for currency in currencies:

        # Subset the dataset for columns containing 'currency'
        currency_data = data[[col for col in data.columns if currency in col]]

        # Compute daily differences
        currency_daily_diff = currency_data.diff().dropna()

        # Normalize the data
        scaler = MinMaxScaler()
        normalized_currency_data = scaler.fit_transform(currency_daily_diff)

        # Convert to PyTorch tensors
        currency_tensor = torch.tensor(normalized_currency_data, dtype=torch.float32)

        # Split data into training and testing sets
        train_size = int(0.8 * len(currency_tensor))
        test_size = len(currency_tensor) - train_size
        train_data, test_data = torch.utils.data.random_split(currency_tensor, [train_size, test_size])

        train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

        # Define the autoencoder architecture (ix32)(32x4)(4x32)(32xi)
        class Autoencoder(nn.Module):
            def __init__(self, input_dim):
                super(Autoencoder, self).__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(input_dim, 32),
                    nn.ReLU(),
                    nn.Linear(32, 4),
                    nn.ReLU()
                )
                self.decoder = nn.Sequential(
                    nn.Linear(4, 32),
                    nn.ReLU(),
                    nn.Linear(32, input_dim),
                    nn.Sigmoid()
                )

            def forward(self, x):
                encoded = self.encoder(x)
                decoded = self.decoder(encoded)
                return decoded

        # Instantiate the autoencoder
        input_dim = currency_tensor.shape[1]
        autoencoder = Autoencoder(input_dim)

        # Define the optimizer and loss function
        optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        # Train the autoencoder
        num_epochs = 50
        for epoch in range(num_epochs):
            autoencoder.train()
            train_loss = 0
            for batch in train_loader:
                optimizer.zero_grad()
                reconstructed = autoencoder(batch)
                loss = criterion(reconstructed, batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)
            if epoch % 10 == 0:
                print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {train_loss:.6f}")

        # Compute reconstruction errors on the entire dataset
        autoencoder.eval()
        with torch.no_grad():
            reconstructed_currency_data = autoencoder(currency_tensor)
            reconstruction_errors = torch.mean((currency_tensor - reconstructed_currency_data) ** 2, dim=1).numpy()

        # # Identify anomalies (reconstruction error > 95th percentile)
        # threshold = np.percentile(reconstruction_errors, 95)
        # anomalies_indices = np.where(reconstruction_errors > threshold)[0]

        # # Extract anomalies and add reconstruction errors
        # anomalies = currency_daily_diff.iloc[anomalies_indices].copy()
        # anomalies['Reconstruction_Error'] = reconstruction_errors[anomalies_indices]

        # # Calculate feature contributions based on reconstruction differences
        # contributions = np.abs(currency_tensor.numpy() - reconstructed_currency_data.numpy())
        # feature_contributions = [dict(zip(currency_daily_diff.columns, contrib)) for contrib in contributions[anomalies_indices]]

        # # Add feature contributions to the anomalies dataframe
        # anomalies['Feature_Contributions'] = feature_contributions

        # # Prepare the final anomalies table
        # result_table = anomalies.reset_index()[['Date', 'Reconstruction_Error', 'Feature_Contributions']]

        # results_overall[currency] = result_table.sort_values('Date')
        
        # Determine the anomaly threshold
        threshold = np.percentile(reconstruction_errors, 95)
        # Identify anomalies (reconstruction error > 95th percentile)
        is_anomaly = reconstruction_errors > threshold

        # Extract all data with reconstruction errors and anomaly flags
        all_data = currency_daily_diff.copy()
        all_data['Anomaly_Score'] = reconstruction_errors
        all_data['Is_Anomaly'] = is_anomaly

        # Calculate feature contributions based on reconstruction differences
        contributions = np.abs(currency_tensor.numpy() - reconstructed_currency_data.numpy())
        feature_contributions = [dict(zip(currency_daily_diff.columns, contrib)) for contrib in contributions]

        # Add feature contributions to the DataFrame
        all_data['Feature_Contributions'] = feature_contributions

        # Prepare the final results table
        result_table = all_data.reset_index()[['Date', 'Anomaly_Score', 'Is_Anomaly', 'Feature_Contributions']]

        # Save results to the dictionary
        results_overall[currency] = result_table.sort_values('Date')
    
    return results_overall

def run_model_skl(file_path = 'C:/Users/dgwal/autoencoder-sf/autoencoder/fixings_SmartXM_no empty_max indices.csv'):
    
    """
    Function to load data from .csv file and run 
    autoencoder for selection of currencies. Outputs 
    a dictionary of results.
    """
    torch.manual_seed(0)

    results_overall = {}

    # Load the dataset
    #file_path = 'C:/Users/dgwal/autoencoder-sf/autoencoder/fixings_SmartXM_no empty_max indices.csv'
    data = pd.read_csv(file_path)

    # Preprocessing: Convert 'Date' to datetime and set it as index
    data['Date'] = pd.to_datetime(data['Date'], format='%d-%b-%y')
    data.set_index('Date', inplace=True)

    # Compute daily differences
    data_daily_diff = data.diff().dropna()

    results_overall['data'] = data
    results_overall['data_diff'] = data_daily_diff
    
    
    currencies = ['JPY', 'EUR', 'USD', 'AUD', 'NZD', 'CAD', 'GBP']
    
    def compute_mahalanobis(data):
        """
        Computes Mahalanobis distance for each point in the dataset.
        """
        mean_vec = np.mean(data, axis=0)
        cov_matrix = np.cov(data, rowvar=False)
        inv_cov_matrix = np.linalg.inv(cov_matrix)
        distances = [mahalanobis(row, mean_vec, inv_cov_matrix) for row in data]
        return distances

    # Set the detection method here
    method = "IsolationForest"  # Options: 'IsolationForest', 'KNN', 'LOF', 'OCSVM', 'Mahalanobis'

    #results_overall = {}  # Dictionary to store results for each currency

    for currency in currencies:
        # Subset the dataset for columns containing 'currency'
        currency_data = data[[col for col in data.columns if currency in col]]

        # Compute daily differences
        currency_daily_diff = currency_data.diff().dropna()

        # Normalize the data
        scaler = MinMaxScaler()
        normalized_currency_data = scaler.fit_transform(currency_daily_diff)

        # Compute anomaly scores based on the chosen method
        if method == "IsolationForest":
            model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
            model.fit(normalized_currency_data)
            anomaly_scores = -model.decision_function(normalized_currency_data)
        elif method == "KNN":
            k = 5  # Number of neighbors
            nbrs = NearestNeighbors(n_neighbors=k).fit(normalized_currency_data)
            distances, _ = nbrs.kneighbors(normalized_currency_data)
            anomaly_scores = distances.mean(axis=1)  # Mean distance to k-nearest neighbors
        elif method == "LOF":
            model = LocalOutlierFactor(n_neighbors=20, contamination=0.05, novelty=True)
            model.fit(normalized_currency_data)
            anomaly_scores = -model.decision_function(normalized_currency_data)
        elif method == "OCSVM":
            model = OneClassSVM(kernel='rbf', gamma='scale', nu=0.05)
            model.fit(normalized_currency_data)
            anomaly_scores = -model.decision_function(normalized_currency_data)
        elif method == "Mahalanobis":
            anomaly_scores = compute_mahalanobis(normalized_currency_data)
        else:
            raise ValueError(f"Unknown method: {method}. Choose from 'IsolationForest', 'KNN', 'LOF', 'OCSVM', 'Mahalanobis'.")

        # Determine the anomaly threshold
        threshold = np.percentile(anomaly_scores, 95)

        # Flag whether each point is an anomaly
        is_anomaly = anomaly_scores > threshold

        # Calculate feature contributions (normalized deviations from the mean for all points)
        feature_contributions = normalized_currency_data - np.mean(normalized_currency_data, axis=0)
        feature_contributions = np.abs(feature_contributions)  # Absolute deviations for interpretability

        # Convert feature contributions to dictionaries
        feature_contributions_dicts = [
            dict(zip(currency_daily_diff.columns, contrib)) for contrib in feature_contributions
        ]

        # Prepare the final results table
        all_data = currency_daily_diff.copy()
        all_data['Anomaly_Score'] = anomaly_scores
        all_data['Is_Anomaly'] = is_anomaly  # Add the anomaly flag column
        all_data['Feature_Contributions'] = feature_contributions_dicts

        # Reset index and prepare the final table
        result_table = all_data.reset_index()[['Date', 'Anomaly_Score', 'Is_Anomaly', 'Feature_Contributions']]

        # Save results to the dictionary
        results_overall[currency] = result_table.sort_values('Date')
        
    return results_overall


def plot_contributions(results, currency):
    
    fc_list = []
    for item in results[currency]['Feature_Contributions']:
         fc_list.append(item)
         
    df = pd.DataFrame(fc_list)

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle('Contributions to reconstruction error from underlying indices')

    # Plot the main line for the feature
    for columns in df:
        plt.plot(df.index, df[columns], label=columns, alpha=0.7, linewidth=1.5)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    return plt
    
    


def plot_results_once(plot_data, currency, anomalies):

    cols = [col for col in plot_data.columns if currency in col]
    currency_data = plot_data[cols]

    fig, axs = plt.subplots(len(cols), 1, figsize=(12, 6 * len(cols)), sharex=True)

    for i, feature in enumerate(cols):

        # Plot the main line for the feature
        axs[i].plot(currency_data.index, currency_data[feature], label=feature, color='steelblue', alpha=0.7, linewidth=1.5)
        fig.suptitle(f"\n{currency} fixings vs Date with Anomalies Highlighted", fontsize=14, fontweight='bold', y=1.02)
        #fig.subplots_adjust(top=0.6)
        
        # Highlight anomalies
        #anomalies = results[currency]
        anomaly_dates = anomalies['Date']
        anomaly_values = currency_data.loc[anomaly_dates, feature]
        axs[i].scatter(anomaly_dates, anomaly_values, color='darkred', label='Anomalies', zorder=5, s=30, alpha=0.8)

        # Add chart aesthetics
        #plt.title(f"{currency} fixings vs Date with Anomalies Highlighted", fontsize=14, fontweight='bold', y=0.98)
        plt.xlabel("Date", fontsize=12)
        #plt.ylabel(feature, fontsize=12)
        axs[i].legend()
        axs[i].grid(alpha=0.3)
        plt.tight_layout()

    return plt


def plot_anomaly_comparison(results_dict: dict, currencies: list):
    # Not really sure if this is a good measure of how 'anomalous' the data is
    overall_anomaly_score = {}

    for curr in currencies:
        overall_anomaly_score[curr] = results_dict[curr]['Anomaly_Score'].mean()
        
        # Plot the 'overall' anomaly scores to get an idea about relative behaviour of each currency

        plt.bar(range(len(overall_anomaly_score)), list(overall_anomaly_score.values()), align='center')
        plt.xticks(range(len(overall_anomaly_score)), list(overall_anomaly_score.keys()))
        plt.xlabel('Currency')
        plt.ylabel('Overall Anomaly Score')
        plt.title('Overall Anomaly Score for Each Currency (mean of Anomaly Score)')
        
    return plt


def plot_anomaly_breakdown(currency, results_overall):
    """
    Plots the primary feature's anomaly scores as a line chart and each feature contribution as separate bar plots,
    all sharing the same x-axis.

    Parameters:
    - currency: The currency to plot (key from `results_overall` dictionary).
    - feature_name: The primary feature to plot.
    - results_overall: Dictionary containing anomaly results for each currency.
    """
    # Retrieve the results for the specified currency
    results = results_overall.get(currency)
    if results is None:
        print(f"No results found for currency: {currency}")
        return

    # Extract the data
    date = results['Date']
    anomaly_scores = results['Anomaly_Score']  # Anomaly scores are plotted on the top axis
    feature_contributions = pd.DataFrame(results['Feature_Contributions'].tolist())

        # Number of features
    n_features = feature_contributions.shape[1]

    # Create subplots: one for anomaly scores and one for each feature's contribution
    fig, axes = plt.subplots(
        n_features, 1, figsize=(16, 2.5 * (n_features + 1)), sharex=True,
        gridspec_kw={"hspace": 0.3}
    )

    
    # Plot each feature's contribution as a separate bar plot
    colors = plt.cm.viridis(np.linspace(0, 1, n_features))
    for i, column in enumerate(feature_contributions.columns):
        axes[i].bar(
            date,
            feature_contributions[column],
            label=column,
            color=colors[i],
            alpha=0.8,
            width=0.8,
        )
        axes[i].set_ylabel(column, fontsize=10)
        axes[i].tick_params(axis="y", labelsize=10)
        axes[i].grid(visible=True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
        axes[i].legend(loc="upper left", fontsize=9)

    # Set x-axis label on the last plot
    axes[-1].set_xlabel("Date", fontsize=12)

    # Set the main title for the figure
    fig.suptitle(f"{currency}: {currency} Anomaly Score and Contributions", fontsize=14)

    # Adjust layout to prevent overlap
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    return plt
