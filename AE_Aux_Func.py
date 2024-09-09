'''
AutoEncoder Auxiliary Functions
'''

import numpy as np
import random
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
import os


# Dimensionality reduction function
def reduce_dimensions(X, method='PCA'):
    if method == 'PCA':
        reducer = PCA(n_components=2)
    elif method == 't-SNE':
        reducer = TSNE(n_components=2, random_state=42)
    else:
        raise ValueError("Method should be 'PCA' or 't-SNE'")
    
    X_reduced = reducer.fit_transform(X)
    return X_reduced


# Function to plot the reduced data
def plot_reduced_data(X_reduced, y, y_pred=None, title="2D Map of Samples"):
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_reduced[:, 0], X_reduced[:, 1], 
                          c=y, cmap='coolwarm', alpha=0.6, edgecolors='w', s=50, label='True Label')
    if y_pred is not None:
        y_pred = np.ravel(y_pred)
        plt.scatter(X_reduced[y_pred == 1, 0], X_reduced[y_pred == 1, 1], 
                    c='red', marker='x', s=100, label='Detected Anomalies')
    plt.colorbar(scatter, label='True Label')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.title(title)
    plt.legend(loc='best')
    plt.show()

def plot_metrics_vs_threshold(thresholds, f1_scores_test, accuracy_test ,precisions_test, recalls_test, roc_aucs_test, optimal_threshold):
    #f1_scores_train, accuracy_train ,precision_train, recalls_train, roc_aucs_train,
    fig = go.Figure()


    # Add traces for test metrics
    fig.add_trace(go.Scatter(x=thresholds, y=f1_scores_test, mode='lines', name='F1 Score_Test'))
    fig.add_trace(go.Scatter(x=thresholds, y=accuracy_test, mode='lines', name='Accuracy_Test'))
    fig.add_trace(go.Scatter(x=thresholds, y=precisions_test, mode='lines', name='Precision_Test'))
    fig.add_trace(go.Scatter(x=thresholds, y=recalls_test, mode='lines', name='Recall_Test'))
    fig.add_trace(go.Scatter(x=thresholds, y=roc_aucs_test, mode='lines', name='ROC-AUC_Test'))

    # Add vertical line for optimal threshold
    fig.add_vline(x=optimal_threshold, line=dict(color='red', width=2, dash='dash'),
                  annotation_text='Optimal Threshold', annotation_position='top right')

    # Update layout to limit the X-axis range
    fig.update_layout(
        title='Metrics vs. Threshold',
        xaxis_title='Threshold',
        yaxis_title='Score',
        legend_title='Metric',
        template='plotly_dark',
        showlegend=True,
        #xaxis_range=[0, 10]  # Limit the X-axis to the specified maximum threshold
    )

    # Show the figure
    fig.show()
    
    
# Function to find the optimal threshold for reconstruction error
def find_optimal_threshold_f1(reconstruction_error, y_true):
    thresholds = np.linspace(min(reconstruction_error), max(reconstruction_error), 100)
    best_threshold = 0
    best_f1 = 0
    for threshold in thresholds:
        y_pred = (reconstruction_error > threshold).astype(int)
        f1 = f1_score(y_true, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    return best_threshold

def find_optimal_threshold_cost_based(reconstruction_error, y_true, cost_fp=2, cost_fn=0.5):
    thresholds = np.linspace(min(reconstruction_error), max(reconstruction_error), 100)
    best_threshold = 0
    best_cost = float('inf')
    
    for threshold in thresholds:
        y_pred = (reconstruction_error > threshold).astype(int)
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        cost = cost_fp * fp + cost_fn * fn
        if cost < best_cost:
            best_cost = cost
            best_threshold = threshold
    
    return best_threshold

def plot_precision_recall_curve(precisions, recalls, thresholds):
    plt.figure(figsize=(8, 6))
    plt.plot(recalls, precisions, marker='.', label="Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.grid(True)
    
    # Mark the optimal threshold
    optimal_idx = np.argmax(2 * (precisions * recalls) / (precisions + recalls))
    optimal_threshold = thresholds[optimal_idx] if thresholds.size > 0 else 0
    plt.scatter(recalls[optimal_idx], precisions[optimal_idx], marker='o', color='red', label=f"Optimal Threshold: {optimal_threshold:.2f}")
    plt.legend()
    
    plt.show()
    
def plot_histogram(erros, bins):
    plt.hist(erros, bins)
    plt.xlabel('Reconstruction Error')
    plt.ylabel('Frequency')
    plt.title('Histogram of Reconstruction Errors')
    plt.show()
    
def plot_reconstruted_data(n_indices, X, X_pred):

    indices = random.sample(range(len(X)), n_indices)

    for i in indices:
        plt.figure(figsize=(12, 4))

        # Original
        plt.subplot(1, 2, 1)
        plt.plot(X[i], label='Original')
        plt.legend()
        plt.title('Original Data')

        # Reconstructed
        plt.subplot(1, 2, 2)
        plt.plot(X_pred[i], label='Reconstructed')
        plt.legend()
        plt.title('Reconstructed Data')

        plt.show()
        


def save_metrics_to_csv(out_dir, file_name, precision, recall, f1, accuracy, auc, optimal_threshold, error_regular, 
                        std_error_regular, error_anomaly, std_error_anomaly, infer_time, proc_time, threshold_metrics=None):
    """
    Save evaluation metrics to a CSV file with metric names in the first column and their values in the second column.
    
    Parameters:
    - out_dir: The directory where the CSV file will be saved.
    - file_name: The name of the CSV file to save the metrics.
    - precision: Precision score of the model.
    - recall: Recall score of the model.
    - f1: F1 score of the model.
    - accuracy: Accuracy of the model.
    - auc: AUC score of the model.
    - optimal_threshold: Optimal threshold determined for the model.
    - error_regular: Average reconstruction error for regular data.
    - std_error_regular: Standard deviation of reconstruction error for regular data.
    - error_anomaly: Average reconstruction error for anomaly data.
    - std_error_anomaly: Standard deviation of reconstruction error for anomaly data.
    - infer_time: Inference time in milliseconds.
    - proc_time: Processing time in milliseconds.
    - threshold_metrics: Optional dictionary of metrics for different thresholds (if available).
    """
    
    # Create a list of tuples where each tuple is (metric_name, metric_value)
    metrics = [
        ('Precision', precision),
        ('Recall', recall),
        ('F1 Score', f1),
        ('Accuracy', accuracy),
        ('AUC', auc),
        ('Optimal Threshold', optimal_threshold),
        ('Average Reconstruction Error (Regular)', error_regular),
        ('Std. Dev. of Reconstruction Error (Regular)', std_error_regular),
        ('Average Reconstruction Error (Anomaly)', error_anomaly),
        ('Std. Dev. of Reconstruction Error (Anomaly)', std_error_anomaly),
        ('Inference Time (ms)', infer_time),
        ('Processing Time (ms)', proc_time)
    ]
    
    # If additional threshold metrics are provided, add them to the list
    if threshold_metrics is not None:
        for threshold, metrics_dict in threshold_metrics.items():
            for metric_name, metric_value in metrics_dict.items():
                metrics.append((f'{metric_name} at {threshold}', metric_value))
    
    # Convert the list of tuples to a DataFrame
    metrics_df = pd.DataFrame(metrics, columns=['Metric', 'Value'])
    
    # Ensure the output directory exists
    os.makedirs(out_dir, exist_ok=True)
    
    # Save the DataFrame to a CSV file
    metrics_df.to_csv(os.path.join(out_dir, file_name), index=False)
    
    print(f"Metrics saved to {os.path.join(out_dir, file_name)}")
    