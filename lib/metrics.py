"""
This module provides functions to calculate various 
evaluation metrics for binary classification tasks.
"""

import torch


def accuracy(y_hat, y):
    """
    Calculates the accuracy metric for binary classification.

    Args:
        y_hat (torch.Tensor): Predicted labels.
        y (torch.Tensor): True labels.

    Returns:
        torch.Tensor: The accuracy metric value.
    """
    acc_val = torch.sum(torch.eq(torch.round(torch.sigmoid(y_hat)), y))/len(y)/2
    return acc_val


def recall(y_hat, y):
    """
    Calculates the recall metric for binary classification.

    Args:
        y_hat (torch.Tensor): Predicted labels.
        y (torch.Tensor): True labels.

    Returns:
        torch.Tensor: The recall metric value.
    """
    recall_val = torch.sum(torch.eq(torch.round(torch.sigmoid(y_hat)), y))/torch.sum(y)/2
    return recall_val.detach()


def precision(y_hat, y):
    """
    Calculates the precision metric for binary classification.

    Args:
        y_hat (torch.Tensor): Predicted labels.
        y (torch.Tensor): True labels.

    Returns:
        torch.Tensor: The precision metric value.
    """
    pred_labels = torch.round(torch.sigmoid(y_hat))
    precision_val = torch.sum(torch.eq(pred_labels, y)) / torch.sum(pred_labels) / 2
    return precision_val.detach()


def f1(y_hat, y):
    """
    Calculates the F1 score metric for binary classification.

    Args:
        y_hat (torch.Tensor): Predicted labels.
        y (torch.Tensor): True labels.

    Returns:
        torch.Tensor: The F1 score metric value.
    """
    prec = precision(y_hat, y)
    rec = recall(y_hat, y)
    f1_val = 2 * prec * rec / (prec + rec)
    return f1_val.detach()
