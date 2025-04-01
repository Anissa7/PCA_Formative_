# Principal Component Analysis (PCA) Implementation

## Overview
This project implements Principal Component Analysis (PCA) from scratch using Python. PCA is a dimensionality reduction technique that transforms data into a lower-dimensional space while retaining as much variance as possible. The assignment involves:

- Computing covariance, eigenvalues, and eigenvectors.
- Dynamically selecting the number of principal components based on explained variance.
- Optimizing PCA for large datasets.
- Applying the implementation to a real-world dataset (fuel_econ.csv).

## Dataset
The dataset used in this assignment is `fuel_econ.csv`, which contains various attributes related to vehicle fuel economy. The data will be preprocessed and transformed using PCA to analyze key relationships between features.

## Implementation Details
### Task 1: PCA from Scratch
- Compute the covariance matrix of the dataset.
- Perform eigendecomposition to obtain eigenvalues and eigenvectors.
- Sort and select the top principal components.
- Project the data onto the new lower-dimensional space.

### Task 2: Dynamic Eigenvalue Selection
- Calculate the explained variance ratio for each principal component.
- Dynamically choose the optimal number of components based on a cumulative variance threshold (e.g., 95%).

### Task 3: Optimization and Scalability
- Implement efficient computation techniques for handling large datasets.
- Use NumPy functions to speed up matrix operations.
- Benchmark performance to evaluate efficiency.

## Installation and Requirements
Ensure you have the necessary dependencies installed:
```bash
pip install numpy pandas matplotlib seaborn
```

Alternatively, if using Google Colab, the required libraries are pre-installed.

## Usage
1. Clone this repository:
```bash
git clone <your-repo-link>
cd <repo-folder>
```
2. Open the Google Colab notebook provided in the repository.
3. Follow the instructions in the notebook to run the PCA implementation.
4. Replace the sample dataset with `fuel_econ.csv` for real-world application.
5. Run the notebook and ensure all outputs (plots, transformed data) are visible.
6. Submit a link to your completed notebook.

## Results
- The PCA implementation will output the transformed dataset with reduced dimensions.
- The explained variance plot will help determine the optimal number of components.
- Performance benchmarks will showcase the efficiency of the implementation.

## Submission
- Submit a GitHub repository link containing the completed notebook and this README file.

## Author
- Anissa Ouedraogo

For any questions or clarifications, feel free to reach out!

