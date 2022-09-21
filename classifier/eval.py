import pandas as pd
import numpy as np
import json
from os.path import exists
from collections import Counter
from collections import defaultdict

# This function gives the accuracy
# input:
# y_pred : is the prediction in numpy form using one hot encoding
# y : is the actual result in numpy form using one hot encoding
# return:
# a floating point value for the accuracy
def accuracy(y_pred, y):
    return (y_pred == y).mean()

# This function gives the f1_score between the prediction and actual data
# input:
# y_pred : is the prediction in numpy form using one hot encoding
# y : is the actual result
# return:
# a floating point value for the accuracy
def f_score(y_pred, y):
    precision_score = precision(y_pred, y)
    recall_score = recall(y_pred, y)
    if precision_score + recall_score == 0:
        return 0
    result = 2 * ((precision_score * recall_score) / (precision_score + recall_score))
    return result

# This function gives the precision between the prediction and actual data
# input:
# y_pred : is the prediction in numpy form using one hot encoding
# y : is the actual result
# return:
# a floating point value for the accuracy
def precision(y_pred, y):
    true_positive = 0
    false_positive = 0
    for row in range(y_pred.shape[0]):
        for column in range(y_pred.shape[1]):
            if y_pred[row][column] == 1 and y_pred[row][column] == y[row][column]:
                true_positive += 1
            if y_pred[row][column] == 1 and y_pred[row][column] != y[row][column]:
                false_positive += 1
    if true_positive + false_positive == 0:
        return 0
    return true_positive / (true_positive + false_positive)

# This function gives the precision between the prediction and actual data
# input:
# y_pred : is the prediction in numpy form using one hot encoding
# y : is the actual result in numpy form using one hot encoding
# return:
# a floating point value for the accuracy
def recall(y_pred, y):
    true_positive = 0
    false_negative = 0
    for row in range(y.shape[0]):
        for column in range(y.shape[1]):
            if y[row][column] == 1 and y_pred[row][column] == y[row][column]:
                true_positive += 1
            if y[row][column] == 1 and y_pred[row][column] != y[row][column]:
                false_negative += 1
    if true_positive + false_negative == 0:
        return 0
    return true_positive / (true_positive + false_negative)