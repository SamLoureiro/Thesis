'''
AutoEncoder Auxiliary Functions
'''

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score


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

def plot_metrics_vs_threshold(thresholds, f1_scores_test, precisions_test, recalls_test, roc_aucs_test,
                              f1_scores_train, precision_train, recalls_train, roc_aucs_train,
                              optimal_threshold):
    fig = go.Figure()

    # Add traces for test metrics
    fig.add_trace(go.Scatter(x=thresholds, y=f1_scores_test, mode='lines', name='F1 Score_Test'))
    fig.add_trace(go.Scatter(x=thresholds, y=precisions_test, mode='lines', name='Precision_Test'))
    fig.add_trace(go.Scatter(x=thresholds, y=recalls_test, mode='lines', name='Recall_Test'))
    fig.add_trace(go.Scatter(x=thresholds, y=roc_aucs_test, mode='lines', name='ROC-AUC_Test'))

    # Add traces for train metrics
    fig.add_trace(go.Scatter(x=thresholds, y=f1_scores_train, mode='lines', name='F1 Score_Train'))
    fig.add_trace(go.Scatter(x=thresholds, y=precision_train, mode='lines', name='Precision_Train'))
    fig.add_trace(go.Scatter(x=thresholds, y=recalls_train, mode='lines', name='Recall_Train'))
    fig.add_trace(go.Scatter(x=thresholds, y=roc_aucs_train, mode='lines', name='ROC-AUC_Train'))

    # Add vertical line for optimal threshold
    fig.add_vline(x=optimal_threshold, line=dict(color='red', width=2, dash='dash'),
                  annotation_text='Optimal Threshold', annotation_position='top right')

    # Update layout for better visualization
    fig.update_layout(
        title='Metrics vs. Threshold',
        xaxis_title='Threshold',
        yaxis_title='Score',
        legend_title='Metric',
        template='plotly_dark',
        showlegend=True
    )

    # Show the figure
    fig.show()
    
    
# Function to find the optimal threshold for reconstruction error
def find_optimal_threshold(reconstruction_error, y_true):
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
