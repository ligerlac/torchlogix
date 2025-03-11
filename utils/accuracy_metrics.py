from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
                             classification_report)
import torch

def evaluate_model(model, data, mask, num_classes):
    """
    Comprehensive evaluation of the model on the specified mask

    Args:
        model: GCN model
        data: PyG Data object containing the graph
        mask: Boolean mask indicating which nodes to evaluate on
        num_classes: Number of classes in the dataset

    Returns:
        metrics: Dictionary containing various performance metrics
    """
    model.eval() # turns off dropout and batch normalization
    with torch.no_grad():
        # Get predictions
        out = model(data)
        pred = out.argmax(dim=1)

        # Extract true labels and predictions for masked nodes
        y_true = data.y[mask].numpy()
        y_pred = pred[mask].numpy()

        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision_micro': precision_score(y_true, y_pred, average='micro', zero_division=0),
            'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall_micro': recall_score(y_true, y_pred, average='micro', zero_division=0),
            'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_micro': f1_score(y_true, y_pred, average='micro', zero_division=0),
            'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
            'confusion_matrix': confusion_matrix(y_true, y_pred, labels=range(num_classes)),
            'classification_report': classification_report(y_true, y_pred, labels=range(num_classes), zero_division=0)
        }

    return metrics
