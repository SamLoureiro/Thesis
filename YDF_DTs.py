import os
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import ydf
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import config  # Import the config file
import PreProc_Function as ppf
from imblearn.over_sampling import SMOTE

# For Future Use
#from sklearn.feature_selection import SelectKBest, f_classif
#from sklearn.model_selection import GridSearchCV

# Helper function to sort files by the numeric part in the filename
def sort_key(file_path):
    file_name = os.path.basename(file_path)
    numeric_part = ''.join(filter(str.isdigit, file_name))
    return int(numeric_part) if numeric_part else 0

def perform_evaluation_and_prediction(model, X_test, Folder, model_string, all_data, average_pre_proc_time, methods_string, current_dir=os.getcwd()):
    
    print("\nModel Description:")
    model.describe()
    
    evaluation = model.evaluate(X_test)
    
    print(model.benchmark(X_test))

    print("\n")
    print('Test evaluation:')
    print(evaluation)

    test_loss = evaluation.loss
    print(f'Test Loss: {test_loss:.4f}')    
    
    # Predict on the test set
    start_time_test_set = time.time()
    y_pred_probs = model.predict(X_test)
    end_time_test_set = time.time()
    
    # For late, when the dataset is labeled with the features names
    
    #print("\nPredictions Analyzed:")
    #model.analyze_prediction(X_test, sampling=0.1)
    
    print("\nVariable importances keys:")
    print(model.variable_importances().keys())
    
    print("\n10 most important features:")
    print(model.variable_importances()["SUM_SCORE"][:10])
    
    print("\n10 less important features:")
    print(model.variable_importances()["SUM_SCORE"][-10:])
    
    print("\nPredictions:")
    
    # Convert predicted probabilities to binary class labels
    y_pred = (y_pred_probs > 0.5).astype(int).flatten()
    
    # Convert labels to numpy array 
    y_test = X_test['label'].values
    
    # Assuming y_pred_probs contains probabilities of being "Damaged Bearing"
    # 0 = Healthy Bearing, 1 = Damaged Bearing

    # Combine the true labels and predicted probabilities into a DataFrame
    probs_df = pd.DataFrame({
        'True Label': y_test.flatten(),  # Assuming y_test is the true binary labels array
        'Predicted Probability': y_pred_probs.flatten()
    })

    # Map the true labels to their corresponding names for better visualization
    probs_df['Label Name'] = probs_df['True Label'].map({0: 'Healthy Bearing', 1: 'Damaged Bearing'})

    # Set up the figure
    plt.figure(figsize=(12, 6))

    # Define a custom color palette with greenish for Healthy and reddish for Damaged
    palette = ['#66c2a5', '#fc8d62']  # Colors: greenish for Healthy, reddish for Damaged
    label_names = ['Healthy Bearing', 'Damaged Bearing']
    color_map = dict(zip(label_names, palette))

    # Plot histograms with KDE disabled for better control over KDE plots
    sns.histplot(data=probs_df, x='Predicted Probability', hue='Label Name', kde=False, bins=50, palette=color_map, alpha=0.4)

    # Plot KDE curves separately for each label to ensure correct color and labeling
    for label in label_names:
        sns.kdeplot(
            data=probs_df[probs_df['Label Name'] == label],
            x='Predicted Probability',
            color=color_map[label],
            label=f'{label} Curve',
            linewidth=2
        )

    # Customize the plot
    plt.xlabel('Predicted Probability')
    plt.ylabel('Density')
    plt.title('Probability Distribution for Healthy vs. Damaged Bearings')
    plt.axvline(0.5, color='red', linestyle='--', label='Decision Threshold (0.50)')

    # Place the legend inside the plot in the upper right
    plt.legend(title='True Label', loc='upper right')
    plt.show()


    # Calculate additional metrics
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_probs)

    # Count the number of good and damaged bearings in the test set
    unique, counts = np.unique(y_test, return_counts=True)
    label_counts = dict(zip(unique, counts))

    print(f"Number of good bearings (0) in the test set: {label_counts.get(0, 0)}")
    print(f"Number of damaged bearings (1) in the test set: {label_counts.get(1, 0)}")

    # Print the inference time
    inference_time = end_time_test_set - start_time_test_set
    average_inference_time = inference_time / len(y_test)
    print("\n")
    print("Time Metrics:")
    print(f"Average Pre-processing Time: {average_pre_proc_time:.4f} seconds")
    print(f"Inference Time: {inference_time:.4f} seconds")
    print(f"Average Inference Time: {average_inference_time:.4f} seconds")

    print("\n")

    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    
    print("\n")

    # Classification report
    print("Classification Report:")
    target_names = ['HEALTHY', 'DAMAGED']
    print(classification_report(y_test, y_pred, target_names=['DAMAGED', 'HEALTHY']))

    # Confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred, labels=[0, 1])

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()

    # Save the residual plot
    if(config.save_metrics):
        results_plot_folder = os.path.join(current_dir, 'DT_Results', 'NEW_AMR', Folder)
        results_plot_path = os.path.join(current_dir, 'DT_Results', 'NEW_AMR', Folder, model_string + '_conf_matrix_' + methods_string + '.svg')
        os.makedirs(results_plot_folder, exist_ok=True)
        plt.savefig(results_plot_path, format='svg')

    plt.show()

    # Save the classification report

    # Gather all metrics
    metrics_dict = {
        'Metric': ['Precision', 'Recall', 'F1 Score', 'Accuracy', 'ROC AUC', 'Loss',
                'Average Pre-processing Time','Average Inference Time'],
        'Value': [precision, recall, f1, accuracy, roc_auc,test_loss,
                average_pre_proc_time, average_inference_time]
    }
    # Save Metrics
    if(config.save_metrics):
        # Create DataFrame
        metrics_df = pd.DataFrame(metrics_dict)

        # Save DataFrame to CSV
        metrics_save_folder = os.path.join(current_dir, 'DT_Results', 'NEW_AMR', Folder)
        metrics_save_path = os.path.join(current_dir, 'DT_Results', 'NEW_AMR', Folder, model_string + '_metrics_' + methods_string + '.csv')
        os.makedirs(metrics_save_folder, exist_ok=True)
        metrics_df.to_csv(metrics_save_path, index=False)

        print("\nMetrics saved to CSV:")
        print(metrics_df)
        
    # Prediction with all data    
    y_pred_probs = model.predict(all_data)
    
    # Convert predicted probabilities to binary class labels
    y_pred = (y_pred_probs > 0.5).astype(int).flatten()
    
    # Convert labels to numpy array 
    y = all_data['label'].values
    
    print("\n")
    
    # Classification report
    print("Classification Report with all the data (train and test):")
    target_names = ['HEALTHY', 'DAMAGED']
    print(classification_report(y, y_pred, target_names=['DAMAGED', 'HEALTHY']))

    # Confusion matrix
    conf_matrix = confusion_matrix(y, y_pred, labels=[0, 1])

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix for All Data')
    plt.tight_layout()
    plt.show()

    return evaluation


# Read data from a csv file
def read_data(csv_file_data, csv_file_labels):
    # Convert preprocessing options to a list of keywords to exclude if they are False
    exclude_keywords = [key for key, value in config.preprocessing_options.items() if not value]
    
    # Read the CSV to get column names
    all_columns = pd.read_csv(csv_file_data, nrows=0).columns

    # Filter out columns that contain any of the keywords with False values
    filtered_columns = [col for col in all_columns if not any(keyword in col for keyword in exclude_keywords)]

    # Read the filtered columns from the CSV file
    df = pd.read_csv(csv_file_data, usecols=filtered_columns)
    
    # Read the labels from the CSV file
    labels = pd.read_csv(csv_file_labels)

    print(df.head())
    
    return df, labels


# Define directories and file paths
def define_directories():
    current_dir = os.getcwd()
    directories = {
        'good_bearing': {
            'audio_m': os.path.join(current_dir, 'Dataset_Bearings', 'AMR_MOVEMENT', 'GOOD', 'AUDIO'),
            'acel_m': os.path.join(current_dir, 'Dataset_Bearings', 'AMR_MOVEMENT', 'GOOD', 'ACEL'),
            'audio_s': os.path.join(current_dir, 'Dataset_Bearings', 'AMR_STOPPED', 'GOOD', 'AUDIO'),
            'acel_s': os.path.join(current_dir, 'Dataset_Bearings', 'AMR_STOPPED', 'GOOD', 'ACEL'),
            'audio_new_amr': os.path.join(current_dir, 'Dataset_Bearings', 'NEW_AMR', 'GOOD', 'AUDIO'),
            'accel_new_amr': os.path.join(current_dir, 'Dataset_Bearings', 'NEW_AMR', 'GOOD', 'ACCEL'),
        },
        'damaged_bearing': {
            'audio_s': os.path.join(current_dir, 'Dataset_Bearings', 'AMR_STOPPED', 'DAMAGED', 'AUDIO'),
            'acel_s': os.path.join(current_dir, 'Dataset_Bearings', 'AMR_STOPPED', 'DAMAGED', 'ACEL'),
            'audio_m': os.path.join(current_dir, 'Dataset_Bearings', 'AMR_MOVEMENT', 'DAMAGED', 'AUDIO'),
            'acel_m': os.path.join(current_dir, 'Dataset_Bearings', 'AMR_MOVEMENT', 'DAMAGED', 'ACEL'),
            'audio_new_amr': os.path.join(current_dir, 'Dataset_Bearings', 'NEW_AMR', 'DAMAGED', 'AUDIO'),
            'accel_new_amr': os.path.join(current_dir, 'Dataset_Bearings', 'NEW_AMR', 'DAMAGED', 'ACCEL'),
        },
        'noise_profile': os.path.join(current_dir, 'Dataset_Piso', 'Noise.WAV')
    }
    return directories


# Helper function to sort files by numeric part in the filename
def sort_key(file_path):
    file_name = os.path.basename(file_path)
    numeric_part = ''.join(filter(str.isdigit, file_name))
    return int(numeric_part) if numeric_part else 0


# Load files and extract features
def load_and_extract_features(directories):
    combined_features = []
    labels = []

    methods_string = "_".join(method for method, value in config.preprocessing_options.items() if value)
    start_time_pre_proc = time.time()

    # Process good bearings
    for audio_file, accel_file in zip(
        sorted([os.path.join(directories['good_bearing']['audio_s'], file) for file in os.listdir(directories['good_bearing']['audio_s']) if file.endswith('.WAV')], key=sort_key) +
        sorted([os.path.join(directories['good_bearing']['audio_m'], file) for file in os.listdir(directories['good_bearing']['audio_m']) if file.endswith('.WAV')], key=sort_key) + 
        sorted([os.path.join(directories['good_bearing']['audio_new_amr'], file) for file in os.listdir(directories['good_bearing']['audio_new_amr']) if file.endswith('.WAV')], key=sort_key),
        sorted([os.path.join(directories['good_bearing']['acel_s'], file) for file in os.listdir(directories['good_bearing']['acel_s']) if file.endswith('.csv')], key=sort_key) +
        sorted([os.path.join(directories['good_bearing']['acel_m'], file) for file in os.listdir(directories['good_bearing']['acel_m']) if file.endswith('.csv')], key=sort_key) +
        sorted([os.path.join(directories['good_bearing']['accel_new_amr'], file) for file in os.listdir(directories['good_bearing']['accel_new_amr']) if file.endswith('.csv')], key=sort_key)
    ):
        audio_features = ppf.extract_audio_features(audio_file, directories['noise_profile'], config.preprocessing_options)
        accel_features = ppf.extract_accel_features(accel_file)
        combined = {**audio_features, **accel_features}
        combined_features.append(combined)
        labels.append(0)  # 0 for good bearing

    n_samples_good_bearing = len(combined_features)
    print(f"Number of samples (Good Bearing): {n_samples_good_bearing}")

    # Process damaged bearings
    for audio_file, accel_file in zip(
        sorted([os.path.join(directories['damaged_bearing']['audio_s'], file) for file in os.listdir(directories['damaged_bearing']['audio_s']) if file.endswith('.WAV')], key=sort_key) +
        sorted([os.path.join(directories['damaged_bearing']['audio_m'], file) for file in os.listdir(directories['damaged_bearing']['audio_m']) if file.endswith('.WAV')], key=sort_key) +
        sorted([os.path.join(directories['damaged_bearing']['audio_new_amr'], file) for file in os.listdir(directories['damaged_bearing']['audio_new_amr']) if file.endswith('.WAV')], key=sort_key),
        sorted([os.path.join(directories['damaged_bearing']['acel_s'], file) for file in os.listdir(directories['damaged_bearing']['acel_s']) if file.endswith('.csv')], key=sort_key) +
        sorted([os.path.join(directories['damaged_bearing']['acel_m'], file) for file in os.listdir(directories['damaged_bearing']['acel_m']) if file.endswith('.csv')], key=sort_key) +
        sorted([os.path.join(directories['damaged_bearing']['accel_new_amr'], file) for file in os.listdir(directories['damaged_bearing']['accel_new_amr']) if file.endswith('.csv')], key=sort_key)
    ):
        audio_features = ppf.extract_audio_features(audio_file, directories['noise_profile'], config.preprocessing_options)
        accel_features = ppf.extract_accel_features(accel_file)
        combined = {**audio_features, **accel_features}
        combined_features.append(combined)
        labels.append(1)  # 1 for damaged bearing

    n_samples_damaged_bearing = len(combined_features) - n_samples_good_bearing
    print(f"Number of samples (Damaged Bearing): {n_samples_damaged_bearing}")

    n_samples_floor_noise = len(combined_features) - n_samples_good_bearing - n_samples_damaged_bearing
    print(f"Number of samples (Floor Noise): {n_samples_floor_noise}")

    end_time_pre_proc = time.time()

    return pd.DataFrame(combined_features), np.array(labels), end_time_pre_proc - start_time_pre_proc, methods_string


def main():
    
    # Current directory
    current_dir = os.getcwd()
    
    ##########################################################################################################################
    
    ## Read from raw data and preprocess
    
    # Load files and extract features
    combined_features_df, y, pre_proc_time, methods_string = load_and_extract_features(define_directories())    

    # Average pre-processing time
    average_pre_proc_time = pre_proc_time / len(y)

    # Normalize features
    scaler = StandardScaler()
    combined_features_normalized = scaler.fit_transform(combined_features_df)

    # Convert features to a DataFrame
    features_df = pd.DataFrame(combined_features_normalized, columns=combined_features_df.columns)
    #features_df.to_csv('features_nr.csv', index=False)

    # Convert labels to a DataFrame
    labels_df = pd.DataFrame(y, columns=["label"])
    #labels_df.to_csv('labels.csv', index=False)
    
    ##########################################################################################################################
    
    ##########################################################################################################################
    
    ## Load preprocessed data from csv files
    
    # Create a string based on the methods that are on
    '''methods_string = "_".join(method for method, value in config.preprocessing_options.items() if value)
    average_pre_proc_time = -1
    
    labels_path = os.path.join(current_dir, 'Dataset_csv', 'Bearings', 'labels.csv')         # Labels
    features_nr = os.path.join(current_dir, 'Dataset_csv', 'Bearings', 'features_nr.csv')    # Features with noise reduction
    features = os.path.join(current_dir, 'Dataset_csv', 'Bearings', 'features.csv')          # Features without noise reduction
    
    if(config.preprocessing_options['noise_reduction']):
        features_df, labels_df = read_data(features_nr, labels_path)
    else:
        features_df, labels_df = read_data(features, labels_path)'''
    

    ##########################################################################################################################
    
    # Oversampling the train dataset using SMOTE
    smt = SMOTE()
    features_df_rs, labels_df_rs = smt.fit_resample(features_df, labels_df)

    # Combine features and labels into one DataFrame
    data = pd.concat([features_df_rs, labels_df_rs], axis=1)

    print(f"Data Shape: {data.shape}")

    # Data Splitting
    X_train, X_test = train_test_split(data, test_size=0.2, random_state=42)


    if(config.model['GBDT']):
        metrics_folder = 'GBDT'
        model_string = 'gbdt'
        model_save_folder = os.path.join(current_dir, "DTs_Models", metrics_folder)
        model_name = model_string + '_' + methods_string
        model_save_path = os.path.join(model_save_folder, model_name)
        if(config.model_load and os.path.exists(model_save_path)):
            print("Model already exists")
            model = ydf.load_model(model_save_path)
        else:
            clf = ydf.GradientBoostedTreesLearner(
                label='label',
                task=ydf.Task.CLASSIFICATION, 
                num_trees=config.model_params_GBDT['num_trees'], 
                growing_strategy=config.model_params_GBDT['growing_strategy'], 
                max_depth=config.model_params_GBDT['max_depth'],
                early_stopping=config.model_params_GBDT['early_stopping'],
                validation_ratio=0.1
            )
            model = clf.train(X_train, verbose=2)
            
            print("\nValidation Results:")
            val_results = model.validation_evaluation()        
            print(val_results)
            # Plot a tree
            model.print_tree(tree_idx=0)

            if(config.save_model and not os.path.exists(model_save_path)):
                os.makedirs(model_save_folder, exist_ok=True)
                model.save(model_save_path)
                
    elif(config.model['RF']):
        metrics_folder = 'RF'
        model_string = 'rf'
        model_save_folder = os.path.join(current_dir, "DTs_Models", metrics_folder)
        model_name = model_string + '_' + methods_string
        model_save_path = os.path.join(model_save_folder, model_name)
        if(config.model_load and os.path.exists(model_save_path)):
            print("Model already exists")
            model = ydf.load_model(model_save_path)
        else:
            clf = ydf.RandomForestLearner(
                label='label',
                task=ydf.Task.CLASSIFICATION, 
                num_trees=config.model_params_GBDT['num_trees'], 
                growing_strategy=config.model_params_GBDT['growing_strategy'], 
                max_depth=config.model_params_GBDT['max_depth'],
            )
            model = clf.train(X_train, verbose=2)
            
            # As the Random Forest model does not require a validation set, we will use the out-of-bag evaluations
            print("\nOut of_bag evaluations:")
            val_results = model.out_of_bag_evaluations()        
            print(val_results)
            # Plot a tree
            model.print_tree(tree_idx=0)

            if(config.save_model and not os.path.exists(model_save_path)):
                os.makedirs(model_save_folder, exist_ok=True)
                model.save(model_save_path)
                
            
    ### perform evaluation and prediction
    perform_evaluation_and_prediction(model, X_test, metrics_folder, model_string, data, average_pre_proc_time, methods_string)
    #model.variable_importances()
    #model.evaluate(X_test)
    # Evaluation caracteristics can be accessed by the code in the comment below
    #evaluation.characteristics[0]. ...    
            
if __name__ == "__main__":
    main()