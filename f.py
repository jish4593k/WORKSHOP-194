import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tkinter import Tk, filedialog

def load_data_gui():
    """Load data interactively using Tkinter GUI."""
    root = Tk()
    root.title("Data Loading GUI")

    file_path = filedialog.askopenfilename(title="Select CSV file", filetypes=[("CSV files", "*.csv")])

    if file_path:
        data = pd.read_csv(file_path)
        root.destroy()
        return data
    else:
        root.destroy()
        return None

def process_star_data(data_list):
    """Clean and process star data."""
    new_data = []
    for star_data in data_list:
        star_data = star_data.replace("''", '').replace(',', '')
        if '-' in star_data:
            val1, val2 = map(float, star_data.split("–"))
            final_value = val1 - val2
        else:
            final_value = float(star_data)
        new_data.append(final_value)
    return new_data

def create_feature_matrix(df):
    """Create feature matrix X."""
    solar_mass = df['Solar Mass (kg)'].tolist()
    solar_radius = df['Solar Radius (m)'].tolist()

    # Sort solar_mass
    solar_mass.sort()

    # Clean and process solar_radius and solar_mass
    new_radius = process_star_data(solar_radius)
    new_mass = process_star_data(solar_mass)

    # Create feature matrix X
    X = np.column_stack((new_radius, new_mass))

    return X

def perform_kmeans_clustering(X):
    """Perform K-Means clustering and plot the elbow method."""
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)

    # Plot the elbow method
    plt.figure(figsize=(10, 5))
    sns.lineplot(range(1, 11), wcss, marker='o', color='red')
    plt.title('The Elbow Method for K-Means Clustering')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Within-Cluster-Sum-of-Squares (WCSS)')
    plt.show()

def main():
    print('Welcome to Advanced Clustering Analysis')

    # Load data interactively using Tkinter GUI
    data = load_data_gui()

    if data is not None:
        # Create feature matrix X
        X = create_feature_matrix(data)

        # Perform K-Means clustering and plot the elbow method
        perform_kmeans_clustering(X)

if __name__ == "__main__":
    main()
