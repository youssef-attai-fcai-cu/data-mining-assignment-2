import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
import random


class K_Means:
    def __init__(self, k=3, max_number_of_iterations=1000):
        self.k = k
        self.max_iter = max_number_of_iterations

    def __random_item(self, data):
        """Choose a random data point as a centroid."""
        return data[random.randint(0, len(data)-1)]

    def __euclidean_distance(self, features, centroid):
        """Calculate the Euclidean distance between a data point and a centroid."""
        return np.linalg.norm(
            features - self.centroids[centroid])

    def fit(self, data):
        """Fit the data to the model."""

        # Initialize centroids
        self.centroids = {}

        # Choose random data points as initial centroids
        for i in range(self.k):
            self.centroids[i] = self.__random_item(data)

        # Optimize centroids by iterating through data points
        for _ in range(self.max_iter):
            # Initialize classes/clusters
            self.classes = {}

            for i in range(self.k):
                self.classes[i] = []

            # Put data points into clusters
            for point in data:
                distances = [self.__euclidean_distance(point, centroid)
                             for centroid in self.centroids]

                smallest_distance = min(distances)

                _class = distances.index(smallest_distance)

                self.classes[_class].append(point)

            # Copy old centroids
            old_centroids = {i: self.centroids[i] for i in self.centroids}

            # Update centroids by taking the average of all points in the cluster
            for _class in self.classes:
                # Check if cluster is not empty
                if self.classes[_class]:
                    self.centroids[_class] = self.__calculate_average(_class)

            # Check for convergence
            optimum = True
            for centroid in self.centroids:
                old_centroid = old_centroids[centroid]
                new_centroid = self.centroids[centroid]

                if self.__converged(old_centroid, new_centroid):
                    optimum = False

            # Break if centroids have converged
            if optimum:
                break

    def __converged(self, old_centroid, new_centroid):
        """Check if the centroids have converged."""
        return np.sum((new_centroid-old_centroid)/old_centroid*100.0) > 0.001

    def predict(self, data):
        """Predict the cluster of a new data point."""
        distances = [np.linalg.norm(data - self.centroids[centroid])
                     for centroid in self.centroids]
        classification = distances.index(min(distances))
        return classification

    def __calculate_average(self, _class):
        """Calculate the average of all points in a cluster."""
        return sum(
            self.classes[_class]) / len(self.classes[_class])


def read_data(filename, percentage):
    """Read data from a CSV file."""
    data = pd.read_csv(filename)
    data = data.sample(frac=percentage, random_state=1)
    return data


def detect_outliers(data):
    """Detect and remove outliers from data."""
    # Detect outliers using IQR
    quantile25 = data.quantile(0.25)
    quantile75 = data.quantile(0.75)
    iqr = quantile75 - quantile25
    outliers = data[(data < (quantile25 - 1.5 * iqr)) |
                    (data > (quantile75 + 1.5 * iqr))]

    # Remove outliers from data
    points_without_outliers = data[~data.isin(outliers)].dropna()

    return points_without_outliers, outliers


def run_clustering(filename, percentage, k):
    """Run clustering on the data."""
    try:
        data = read_data(filename, percentage)

        # Detect and remove outliers
        points_without_outliers, outliers = detect_outliers(
            data['IMDB Rating'])

        # Convert relevant columns to numpy array for clustering
        features = points_without_outliers.values.reshape(-1, 1)

        k_means = K_Means(k=k)
        k_means.fit(features)

        # Output cluster contents
        results = {}
        for i in range(k):
            cluster_results = [row for j, row in enumerate(
                points_without_outliers.values) if k_means.predict(features[j]) == i]
            results[f"Cluster {i+1}"] = cluster_results

        return results, outliers

    except Exception as e:
        print("Something went wrong :(")
        print(e)
        return None, None


class GUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("IMDb Movies Clustering")

        self.csv_path_label = tk.Label(self.root, text="CSV Path:")
        self.csv_path_label.grid(row=0, column=0)

        self.csv_path_entry = tk.Entry(self.root, width=30)
        self.csv_path_entry.grid(row=0, column=1)

        self.choose_csv_button = tk.Button(
            self.root, text="Choose CSV", command=lambda: self.choose_csv())
        self.choose_csv_button.grid(row=0, column=2)

        self.percentage_label = tk.Label(self.root, text="Percentage (0-100):")
        self.percentage_label.grid(row=1, column=0)
        self.percentage_entry = tk.Entry(self.root, width=10)
        self.percentage_entry.grid(row=1, column=1)

        self.k_label = tk.Label(self.root, text="k:")
        self.k_label.grid(row=2, column=0)
        self.k_entry = tk.Entry(self.root, width=10)
        self.k_entry.grid(row=2, column=1)

        self.run_button = tk.Button(
            self.root, text="Run", command=lambda: self.run())
        self.run_button.grid(row=3, column=1)

        self.root.mainloop()

    def choose_csv(self):
        filename = filedialog.askopenfilename(
            filetypes=[("CSV files", "*.csv")])
        self.csv_path_entry.delete(0, tk.END)
        self.csv_path_entry.insert(0, filename)

    def run(self):
        try:
            percentage = float(self.percentage_entry.get())
            percentage /= 100

            k = int(self.k_entry.get())

            cluster_contents, outliers = run_clustering(
                self.csv_path_entry.get(), percentage, k)
            output_window = tk.Toplevel()
            output_window.title("Clustering Output")

            # Create a Text widget to display cluster contents
            cluster_text = tk.Text(output_window, width=50, height=20)
            cluster_text.pack(fill=tk.BOTH, expand=True)

            # Output cluster contents
            cluster_text.insert(tk.END, "Cluster contents:\n\n")
            for cluster, contents in cluster_contents.items():
                cluster_text.insert(tk.END, f"{cluster}:\n")
                for content in contents:
                    cluster_text.insert(tk.END, f"{content}\n")
                cluster_text.insert(tk.END, "\n")

            # Create a Text widget to display outliers
            outliers_text = tk.Text(output_window, width=50, height=5)
            outliers_text.pack(fill=tk.BOTH, expand=True)

            # Output outliers
            outliers_text.insert(tk.END, "Outliers:\n\n")
            outliers_text.insert(tk.END, outliers.to_string(index=False))

        except ValueError:
            messagebox.showerror(
                "Error", "Please enter valid input for percentage and k.")

    def main(self):
        self.root.mainloop()


if __name__ == "__main__":
    # main()
    gui = GUI()
    gui.main()
