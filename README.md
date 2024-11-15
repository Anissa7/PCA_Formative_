Public Health Risk Forecasting in Burkina Faso
Introduction
This project focuses on predicting regions at high risk of public health crises based on health and sanitation data in Burkina Faso. The aim is to use machine learning classification techniques to anticipate health risks and enable preventive measures in vulnerable areas. This analysis can provide valuable insights for government and healthcare stakeholders, aiding in the formulation of strategies to address public health challenges.

The project will involve:

Cleaning and preparing a dataset of health indicators in Burkina Faso.
Building a vanilla model as a baseline for public health risk classification.
Applying optimization techniques to improve model performance.
Conducting error analysis and evaluating model results.
Dataset
Overview
The dataset used for this project includes health indicators from Burkina Faso. Each entry represents a health-related measure for a specific year, including factors such as health infrastructure, disease prevalence, and access to sanitation.

Key Features:
GHO (CODE): The code for each health indicator.
GHO (DISPLAY): A description of the health indicator.
YEAR (DISPLAY): The year the data was recorded.
COUNTRY (DISPLAY): The country for which the data applies (Burkina Faso).
Value: The numerical value representing the indicator for that year.
Data Cleaning
Removed irrelevant columns (e.g., URLs, metadata) and rows that contained missing values.
Ensured that the "Value" column, which contains the health indicator measurements, is numeric for modeling purposes.
The cleaned dataset is saved as health_indicators_clean.csv and used for model training.

Machine Learning Models
Vanilla Model (Baseline Model)
The first model is a vanilla machine learning classifier, which serves as the baseline for this project. It’s a simple model without any optimizations or regularization. This model helps establish the baseline performance of the dataset.

Model Type: A basic logistic regression or decision tree classifier.
Training: The model is trained without any tuning or advanced techniques.
Purpose: The vanilla model is used to compare improvements made by the optimized model.
The model is saved as vanilla_model.h5.

Optimized Model
The optimized model incorporates the following techniques to improve performance:

Regularization (L1 or L2): Helps to prevent overfitting by penalizing large coefficients in the model.
Learning Rate Adjustment: Dynamically adjusts the learning rate to speed up convergence during training.
Early Stopping: Halts training when the model’s performance on the validation set stops improving, avoiding overfitting.
Hyperparameter Tuning: Optimizes parameters like the number of layers, learning rate, and batch size to achieve the best performance.
The optimized model is saved as optimized_model.h5.

Optimization Techniques
We implemented several optimization methods to enhance model performance:

Regularization: Applied to the model to reduce overfitting and improve generalization to unseen data.
Learning Rate Adjustment: Adjusted during training to find the best balance between convergence speed and accuracy.
Early Stopping: Implemented to prevent overfitting by stopping training when the validation loss starts increasing.
Parameters:
Regularization (L2): 
𝜆
=
0.01
λ=0.01
Initial Learning Rate: 
0.001
0.001
Batch Size: 
32
32
Error Analysis
We conducted a comprehensive error analysis to evaluate the models. Metrics used for the analysis include:

Specificity: Measures the proportion of true negatives that are correctly identified.
Confusion Matrix: Provides insight into the model’s predictions and errors.
F1 Score: Balances precision and recall to provide a single performance metric.
Metrics and Evaluation:
Validation Accuracy: Measures the model’s ability to generalize to unseen data.
F1 Score: Focuses on balancing false positives and false negatives, especially important for imbalanced datasets.
The confusion matrix and detailed metrics are saved in the results/ folder.

Performance Baseline
Model Comparison:
The difference in validation accuracy between the vanilla model and the optimized model is over 3%, demonstrating significant improvement.
Both models show consistent results, with the optimized model reaching an accuracy above 80%, significantly outperforming the vanilla model.
Conclusion
This project successfully implemented machine learning techniques to predict public health risks in Burkina Faso. The optimized model, with regularization, early stopping, and learning rate adjustments, showed a significant improvement over the vanilla model. This analysis can help decision-makers identify regions at high risk and allocate resources more effectively.

How to Run the Project
Requirements:
Python 3.x
TensorFlow/Keras
Pandas, Numpy
Matplotlib, Seaborn
Steps to Run:
Clone the repository.
Install dependencies: pip install -r requirements.txt
Run the data_cleaning.ipynb notebook to load and preprocess the data.
Train the models using model_training.ipynb.
Review the results and error analysis in the results/ folder.
