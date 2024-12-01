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



def run_model(
    method: str,
    file_path: str = 'C:/Users/dgwal/autoencoder-sf/autoencoder/fixings_SmartXM_no empty_max indices.csv'
):
    """
    Loads data, preprocesses it, and performs anomaly detection using either a PyTorch-based autoencoder
    or sklearn-based models for a set of currencies.
    
    Parameters:
        method (str): The anomaly detection method. Options include:
            - "Autoencoder" for PyTorch-based autoencoder
            - "IsolationForest", "KNN", "LOF", "OCSVM", "Mahalanobis" for sklearn-based methods
        file_path (str): Path to the CSV file containing the dataset.

    Returns:
        dict: A dictionary with results for each currency.
    """

    # Set random seed
    torch.manual_seed(0)

    # Load and preprocess the dataset
    data = pd.read_csv(file_path)
    data['Date'] = pd.to_datetime(data['Date'], format='%d-%b-%y')
    data.set_index('Date', inplace=True)
    data_daily_diff = data.diff().dropna()

    results_overall = {'data': data, 'data_diff': data_daily_diff}

    # Define currencies
    currencies = [
        'AED', 'AUD', 'CAD', 'CHF', 'CNY', 'CZK', 'DKK', 'EUR', 'GBP',
        'HUF', 'JPY', 'NOK', 'NZD', 'PLN', 'RUB', 'SEK', 'SGD', 'THB',
        'TRY', 'USD'
    ]

    # Helper functions
    def compute_mahalanobis(data):
        """Computes Mahalanobis distance for each data point."""
        mean_vec = np.mean(data, axis=0)
        cov_matrix = np.cov(data, rowvar=False)
        inv_cov_matrix = np.linalg.inv(cov_matrix)
        return [mahalanobis(row, mean_vec, inv_cov_matrix) for row in data]

    def train_autoencoder(data, input_dim):
        """Trains a PyTorch autoencoder and computes reconstruction errors."""
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
                return self.decoder(self.encoder(x))

        # Initialize the autoencoder, optimizer, and loss function
        autoencoder = Autoencoder(input_dim=input_dim)
        optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        # Prepare data loaders
        tensor_data = torch.tensor(data, dtype=torch.float32)
        train_size = int(0.8 * len(tensor_data))
        train_data, _ = torch.utils.data.random_split(tensor_data, [train_size, len(tensor_data) - train_size])
        train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

        # Train the autoencoder
        for epoch in range(50):
            autoencoder.train()
            for batch in train_loader:
                optimizer.zero_grad()
                loss = criterion(autoencoder(batch), batch)
                loss.backward()
                optimizer.step()

        # Compute reconstruction errors
        autoencoder.eval()
        with torch.no_grad():
            reconstructed = autoencoder(tensor_data)
            errors = torch.mean((tensor_data - reconstructed) ** 2, dim=1).numpy()

        return errors

    def get_anomaly_scores(method, data):
        """Calculates anomaly scores using the specified method."""
        if method == "IsolationForest":
            model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
            model.fit(data)
            return -model.decision_function(data)
        elif method == "KNN":
            k = 5
            nbrs = NearestNeighbors(n_neighbors=k).fit(data)
            distances, _ = nbrs.kneighbors(data)
            return distances.mean(axis=1)
        elif method == "LOF":
            model = LocalOutlierFactor(n_neighbors=20, contamination=0.05, novelty=True)
            model.fit(data)
            return -model.decision_function(data)
        elif method == "OCSVM":
            model = OneClassSVM(kernel='rbf', gamma='scale', nu=0.05)
            model.fit(data)
            return -model.decision_function(data)
        elif method == "Mahalanobis":
            return compute_mahalanobis(data)
        else:
            raise ValueError(f"Unknown method: {method}. Choose from 'Autoencoder', 'IsolationForest', 'KNN', 'LOF', 'OCSVM', 'Mahalanobis'.")

    # Process each currency
    for currency in currencies:
        # Subset and preprocess the data
        currency_data = data[[col for col in data.columns if currency in col]].diff().dropna()
        normalized_data = MinMaxScaler().fit_transform(currency_data)

        # Get anomaly scores
        if method == "Autoencoder":
            anomaly_scores = train_autoencoder(normalized_data, input_dim=normalized_data.shape[1])
        else:
            anomaly_scores = get_anomaly_scores(method, normalized_data)

        # Determine anomalies
        threshold = np.percentile(anomaly_scores, 95)
        is_anomaly = anomaly_scores > threshold

        # Compute feature contributions
        feature_contributions = np.abs(normalized_data - np.mean(normalized_data, axis=0))
        feature_contributions_dicts = [
            dict(zip(currency_data.columns, contrib)) for contrib in feature_contributions
        ]

        # Prepare results
        all_data = currency_data.copy()
        all_data['Anomaly_Score'] = anomaly_scores
        all_data['Is_Anomaly'] = is_anomaly
        all_data['Feature_Contributions'] = feature_contributions_dicts

        # Save results
        results_overall[currency] = all_data.reset_index()[
            ['Date', 'Anomaly_Score', 'Is_Anomaly', 'Feature_Contributions']
        ].sort_values('Date')

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
        fig.suptitle(f"\n{currency} fixings vs Date with Anomalies Highlighted", fontsize=14, fontweight='bold', y=0.98)
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
        #plt.tight_layout()

    return plt


def plot_anomaly_comparison(results_dict: dict, currencies: list):
    # Not really sure if this is a good measure of how 'anomalous' the data is
    overall_anomaly_score = {}

    for curr in currencies:
        overall_anomaly_score[curr] = results_dict[curr]['Anomaly_Score'].mean()
        
        # Plot the 'overall' anomaly scores to get an idea about relative behaviour of each currency

        plt.bar(range(len(overall_anomaly_score)), list(overall_anomaly_score.values()), align='center')
        plt.xticks(range(len(overall_anomaly_score)), list(overall_anomaly_score.keys()), rotation='vertical')
        plt.xlabel('Currency')
        #plt.xticks(rotation='vertical')
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
            alpha=1,
            width=1,
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
