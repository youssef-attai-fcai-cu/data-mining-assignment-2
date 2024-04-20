import random
import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class K_Means:
    @staticmethod
    def start(filename, percentage, k):
        """Run clustering on the data."""
        try:
            data = CSVReader.read_data(filename, percentage)

            # Detect and remove outliers
            points, outliers = OutlierCalculator.iqr(
                data['IMDB Rating']
            )

            # Convert relevant columns to numpy array for clustering
            features = points.values.reshape(-1, 1)

            k_means = K_Means(k=k)

            # Fit the data (without outliers) to the model
            k_means.fit(features)

            # Output cluster contents
            results = {}
            for i in range(k):
                cluster_results = [jj for j, jj in enumerate(
                    points.values) if k_means.predict(features[j]) == i]

                results[f"Cluster {i+1}"] = cluster_results

            return results, outliers

        except Exception as e:
            print("Something went wrong :(")
            print(e)
            messagebox.showerror("Error", "Something went wrong :(")
            return None, None

    def __init__(self, k=3, max_number_of_iterations=1000):
        self.max_iter = max_number_of_iterations
        self.k = k

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

            # Update centroids (average of all points in the cluster)
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

    def predict(self, data):
        """Predict the cluster of a new data point."""
        distances = [np.linalg.norm(data - self.centroids[centroid])
                     for centroid in self.centroids]
        _class = distances.index(min(distances))
        return _class

    def __converged(self, old_centroid, new_centroid):
        """
        Check if the centroids have converged.

        If the centroids have moved more than 0.001%, they have not converged.
        Otherwise, they have converged.

        Calculated as the sum of the percentage change in each dimension.
        """
        return np.sum((new_centroid-old_centroid)/old_centroid*100.0) > 0.001

    def __calculate_average(self, _class):
        """Calculate the average of all points in a cluster."""
        return sum(
            self.classes[_class]) / len(self.classes[_class])

    def __random_item(self, data):
        """Choose a random data point as a centroid."""
        return data[random.randint(0, len(data)-1)]

    def __euclidean_distance(self, features, centroid):
        """Calculate the Euclidean distance between a point and a centroid."""
        return np.linalg.norm(
            features - self.centroids[centroid])


class CSVReader:
    @staticmethod
    def read_data(filename, percentage):
        """Read data from a CSV file."""
        data = pd.read_csv(filename)
        data = data.sample(frac=percentage, random_state=1)
        return data


class OutlierCalculator:
    @staticmethod
    def z_score(data):
        """Detect and remove outliers from data."""

        # Calculate Z-score
        z = np.abs((data - data.mean()) / data.std())

        # Calculate outliers using Z-score
        outliers = data[z > 3]

        # Remove outliers from data
        data_in_outliers = data.isin(outliers)
        data_not_in_outliers = data[~data_in_outliers]
        data_not_in_outliers_without_nan = data_not_in_outliers.dropna()

        return data_not_in_outliers_without_nan, outliers

    @staticmethod
    def iqr(data):
        """Detect and remove outliers from data."""

        # Calculate IQR
        quantile25 = data.quantile(0.25)
        quantile75 = data.quantile(0.75)
        iqr = quantile75 - quantile25

        # Calculate outliers using IQR
        data_less_than_lower_bound = data < (quantile25 - 1.5 * iqr)
        data_greater_than_upper_bound = data > (quantile75 + 1.5 * iqr)
        outliers = data[data_less_than_lower_bound |
                        data_greater_than_upper_bound]

        # Remove outliers from data
        data_in_outliers = data.isin(outliers)
        data_not_in_outliers = data[~data_in_outliers]
        data_not_in_outliers_without_nan = data_not_in_outliers.dropna()

        return data_not_in_outliers_without_nan, outliers


class GUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("IMDb Movies Clustering")

        # CSV path
        self.csv_path_label = tk.Label(self.root, text="CSV Path:")
        self.csv_path_entry = tk.Entry(self.root, width=30)

        # Choose CSV button
        self.choose_csv_button = tk.Button(
            self.root, text="Choose CSV", command=lambda: self.choose_csv())

        # Percentage of data to use
        self.percentage_label = tk.Label(self.root, text="Percentage (0-100):")
        self.percentage_entry = tk.Entry(self.root, width=10)

        # k value
        self.k_label = tk.Label(self.root, text="k:")
        self.k_entry = tk.Entry(self.root, width=10)

        # Run button
        self.run_button = tk.Button(
            self.root, text="Run", command=lambda: self.run())

        # Matplotlib figure
        self.figure = Figure(figsize=(5, 4), dpi=100)
        self.plot = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, self.root)
        self.canvas.draw()

        # Place widgets
        self.csv_path_label.grid(row=0, column=0)
        self.csv_path_entry.grid(row=0, column=1)
        self.choose_csv_button.grid(row=0, column=2)
        self.percentage_label.grid(row=1, column=0)
        self.percentage_entry.grid(row=1, column=1)
        self.k_label.grid(row=2, column=0)
        self.k_entry.grid(row=2, column=1)
        self.run_button.grid(row=3, column=0)
        self.canvas.get_tk_widget().grid(row=4, column=0)

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

            csv_path = self.csv_path_entry.get()

            clustering_result, outliers = K_Means.start(
                csv_path, percentage, k)

            print(F"Clusters: {clustering_result}")
            print(F"Outliers: {outliers}")

            # Display cluster contents in Matplotlib
            self.plot.clear()
            for cluster, points_in_cluster in clustering_result.items():
                self.plot.scatter(
                    np.arange(len(points_in_cluster)),
                    points_in_cluster,
                    label=cluster)
            self.plot.legend()
            self.canvas.draw()

            # Color outliers red
            if outliers is not None:
                self.plot.scatter(
                    np.arange(len(outliers)),
                    outliers, color='red', label='Outliers'
                )
                self.plot.legend()
                self.canvas.draw()

        except ValueError:
            messagebox.showerror(
                "Error", "Please enter valid input for percentage and k.")

    def main(self):
        self.root.mainloop()


if __name__ == "__main__":
    gui = GUI()
    gui.main()
