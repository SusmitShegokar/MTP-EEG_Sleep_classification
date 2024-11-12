import os
import numpy as np
import torch
from sklearn.metrics import (
    precision_score, recall_score, f1_score, cohen_kappa_score, roc_auc_score, classification_report, confusion_matrix,
    roc_curve, auc, precision_recall_curve
)
import seaborn as sns
import matplotlib.pyplot as plt
import time
import argparse
import importlib.util
from sklearn.preprocessing import label_binarize

# Function to load the model architecture from a given .py file
def load_model_class(model_py_path):
    spec = importlib.util.spec_from_file_location("EEGClassifier", model_py_path)
    model_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_module)
    return model_module.EEGClassifier

# Function to load the saved model
def load_model(model_path, model_class, device='cuda'):
    model = model_class()  # Initialize the model architecture (EEGClassifier in your case)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# Function to load test data from .npy files
def load_test_data():
    test_data = np.load('data/test/data.npy')
    test_labels = np.load('data/test/labels.npy')
    return test_data, test_labels

# Batched inference to reduce memory usage, assuming model outputs probabilities
def batched_inference(model, test_data, batch_size=64, device='cuda'):
    outputs = []
    model.eval()

    for i in range(0, len(test_data), batch_size):
        batch_data = torch.tensor(test_data[i:i+batch_size]).float().to(device)
        with torch.no_grad():
            batch_output = model(batch_data).cpu().numpy()  # Move to CPU to free GPU memory
        outputs.append(batch_output)

    outputs = np.vstack(outputs)  # Combine batches into one array
    return outputs

# Function to evaluate metrics (precision, recall, F1 score, support, etc.)
def evaluate_metrics(test_labels, predictions, outputs, num_classes):
    precision = precision_score(test_labels, predictions, average='weighted')
    recall = recall_score(test_labels, predictions, average='weighted')
    f1 = f1_score(test_labels, predictions, average='weighted')
    kappa = cohen_kappa_score(test_labels, predictions)

    # Calculate AUC-ROC (multi-class)
    try:
        # One-hot encode the labels for multi-class ROC-AUC calculation
        test_labels_bin = label_binarize(test_labels, classes=list(range(num_classes)))
        auc_score = roc_auc_score(test_labels_bin, outputs, multi_class='ovo', average='weighted')
    except ValueError:
        auc_score = None
        print("AUC-ROC could not be calculated due to data formatting.")
    
    # Generate support for each class from classification report
    report_dict = classification_report(test_labels, predictions, output_dict=True)
    support = {label: metrics["support"] for label, metrics in report_dict.items() if label.isdigit()}

    return precision, recall, f1, kappa, auc_score, support

# Function to plot ROC curve for each class and save
def plot_roc_curve(test_labels, outputs, num_classes, model_name):
    test_labels_bin = label_binarize(test_labels, classes=list(range(num_classes)))

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    plt.figure(figsize=(8, 6))

    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(test_labels_bin[:, i], outputs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    roc_filename = f"./output/{model_name}/{model_name}_roc_curve.png"
    plt.savefig(roc_filename)
    print(f"Saved ROC curve as {roc_filename}")
    plt.show()

# Function to plot Precision-Recall curve for each class and save
def plot_pr_curve(test_labels, outputs, num_classes, model_name):
    test_labels_bin = label_binarize(test_labels, classes=list(range(num_classes)))

    precision = dict()
    recall = dict()
    pr_auc = dict()

    plt.figure(figsize=(8, 6))

    for i in range(num_classes):
        precision[i], recall[i], _ = precision_recall_curve(test_labels_bin[:, i], outputs[:, i])
        pr_auc[i] = auc(recall[i], precision[i])
        plt.plot(recall[i], precision[i], label=f'Class {i} (AUC = {pr_auc[i]:.2f})')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower left')
    pr_filename = f"./output/{model_name}/{model_name}_pr_curve.png"
    plt.savefig(pr_filename)
    print(f"Saved Precision-Recall curve as {pr_filename}")
    plt.show()

# Confusion matrix and classification report
def generate_confusion_matrix(test_labels, predictions, model_name):
    conf_matrix = confusion_matrix(test_labels, predictions)
    report = classification_report(test_labels, predictions)

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    conf_matrix_filename = f"./output/models/{model_name}/{model_name}_confusion_matrix.png"
    plt.savefig(conf_matrix_filename)
    print(f"Saved Confusion Matrix as {conf_matrix_filename}")
    plt.show()

    return conf_matrix, report

# Function to save evaluation metrics to a file
def save_results_to_file(precision, recall, f1, kappa, auc_score, support, conf_matrix, report, test_duration, model_name):
    file_name = f"./output/models/{model_name}/{model_name}_eval_result.txt"
    with open(file_name, "w") as f:
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")
        f.write(f"Kappa: {kappa:.4f}\n")
        f.write(f"AUC: {auc_score:.4f}\n" if auc_score else "AUC: Not calculated\n")
        f.write(f"Test Duration: {test_duration:.4f} seconds\n")
        f.write(f"Support (number of true instances per class):\n")
        for label, count in support.items():
            f.write(f"Class {label}: {count} instances\n")
        f.write(f"Confusion Matrix:\n{conf_matrix}\n")
        f.write(f"Classification Report:\n{report}\n")
    print(f"Saved evaluation results as {file_name}")

# Main evaluation function
def evaluate_model_with_timing(model, test_data, test_labels, num_classes=6, batch_size=120, device='cuda', model_name="model"):
    # Start time for inference
    start_time = time.time()

    # Run batched inference
    outputs = batched_inference(model, test_data, batch_size=batch_size, device=device)

    # End time for inference
    end_time = time.time()
    test_duration = end_time - start_time

    # Predictions and ground truth
    predictions = np.argmax(outputs, axis=1)

    # Evaluate metrics (including support)
    precision, recall, f1, kappa, auc_score, support = evaluate_metrics(test_labels, predictions, outputs, num_classes)
    
    # Confusion matrix and classification report
    conf_matrix, report = generate_confusion_matrix(test_labels, predictions, model_name)

    # Plot ROC and PR curves
    plot_roc_curve(test_labels, outputs, num_classes, model_name)
    plot_pr_curve(test_labels, outputs, num_classes, model_name)

    return precision, recall, f1, kappa, auc_score, support, conf_matrix, report, test_duration

# # Main function to load model, test data, and evaluate
def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained model on test data.")
    parser.add_argument('model_path', type=str, help='Path to the .pt file of the trained model')
    parser.add_argument('model_py_path', type=str, help='Path to the .py file of the model architecture')
    args = parser.parse_args()

    # Extract model name from the .py file path (remove .py extension)
    model_name = os.path.basename(args.model_py_path).replace('.py', '')

    # Load model architecture from the provided .py file
    model_class = load_model_class(args.model_py_path)

    # Load the model
    model = load_model(args.model_path, model_class, device='cuda')

    # Load test data
    test_data, test_labels = load_test_data()

    # Evaluate model and compute metrics with timing
    precision, recall, f1, kappa, auc_score, support, conf_matrix, report, test_duration = evaluate_model_with_timing(
        model, test_data, test_labels, num_classes=6, batch_size=64, device='cuda', model_name=model_name
    )

    # Print the metrics to the console
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Kappa: {kappa:.4f}")
    print(f"AUC: {auc_score:.4f}" if auc_score else "AUC: Not calculated")
    print(f"Support: {support}")
    print(f"Confusion Matrix:\n{conf_matrix}")
    print(f"Classification Report:\n{report}")

    # Save the metrics to a file
    save_results_to_file(precision, recall, f1, kappa, auc_score, support, conf_matrix, report, test_duration, model_name)

if __name__ == "__main__":
    main()
