import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Define the confusion matrix data
confusion_matrix = np.array([[4291, 4], [10, 4285]])
target_names = ['NOISE', 'BEARING']
# Create a figure and axis
plt.figure(figsize=(8, 6))

# Use seaborn to create a heatmap for the confusion matrix
sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues", cbar=False, 
            xticklabels=target_names, 
            yticklabels=target_names)

# Add labels and title
plt.xlabel("Prediction")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.savefig('cross_metrics/cf_cross_rf_mfcc.svg', format='svg')
# Show the plot
plt.show()