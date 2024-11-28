import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import streamlit as st

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

        # Identify anomalies (reconstruction error > 95th percentile)
        threshold = np.percentile(reconstruction_errors, 95)
        anomalies_indices = np.where(reconstruction_errors > threshold)[0]

        # Extract anomalies and add reconstruction errors
        anomalies = currency_daily_diff.iloc[anomalies_indices].copy()
        anomalies['Reconstruction_Error'] = reconstruction_errors[anomalies_indices]

        # Calculate feature contributions based on reconstruction differences
        contributions = np.abs(currency_tensor.numpy() - reconstructed_currency_data.numpy())
        feature_contributions = [dict(zip(currency_daily_diff.columns, contrib)) for contrib in contributions[anomalies_indices]]

        # Add feature contributions to the anomalies dataframe
        anomalies['Feature_Contributions'] = feature_contributions

        # Prepare the final anomalies table
        result_table = anomalies.reset_index()[['Date', 'Reconstruction_Error', 'Feature_Contributions']]

        # Display the result table
        #print(result_table.head())

        results_overall[currency] = result_table.sort_values('Date')
    
    return results_overall



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
        overall_anomaly_score[curr] = results_dict[curr]['Reconstruction_Error'].mean()
        
        # Plot the 'overall' anomaly scores to get an idea about relative behaviour of each currency

        plt.bar(range(len(overall_anomaly_score)), list(overall_anomaly_score.values()), align='center')
        plt.xticks(range(len(overall_anomaly_score)), list(overall_anomaly_score.keys()))
        plt.xlabel('Currency')
        plt.ylabel('Overall Anomaly Score')
        plt.title('Overall Anomaly Score for Each Currency (mean of Reconstruction Error)')
        
    return plt