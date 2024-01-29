import torch


def accuracy(y_hat, y):
    acc = torch.sum(torch.eq(torch.round(torch.sigmoid(y_hat)), y))/len(y)/2
    return acc

def recall(y_hat, y):
    recall = torch.sum(torch.eq(torch.round(torch.sigmoid(y_hat)), y))/torch.sum(y)/2
    return recall.detach()

def precision(y_hat, y):
    precision = torch.sum(torch.eq(torch.round(torch.sigmoid(y_hat)), y))/torch.sum(torch.round(torch.sigmoid(y_hat)))/2
    return precision.detach()

def f1(y_hat, y):
    prec = precision(y_hat, y)
    rec = recall(y_hat, y)
    f1 = 2 * prec * rec / (prec + rec)
    return f1.detach()
