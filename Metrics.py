import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.metrics import Precision, Recall, TruePositives, TrueNegatives, FalsePositives, FalseNegatives, Accuracy, BinaryAccuracy
from tensorflow_addons.metrics import F1Score, MultiLabelConfusionMatrix

def F1_score(y_t, y_p, weights):
    """Compute the weighted F1_score for each label
    Argument: ground truth label Y (3D flattened into 2D), prediction (3D flattened into 2D), class weights
    Return: array of F1_score for each label, and weighted F1 score"""

    P = Precision()
    R = Recall() #label per label evaluation
    F1_score_per_label = [] #store per label
    P_per_label = []
    R_per_label = []
    F1_tot = 0 #weighted sum

    for i in range(8):
      P.update_state( y_t[:,i], y_p[:,i] )
      R.update_state( y_t[:,i], y_p[:,i] )
      p = P.result().numpy()
      r = R.result().numpy()
      P.reset_states()
      R.reset_states()
      if p+r == 0:
        f1 = 0
      else:
        f1 = 2*p*r/ (p+r)
      F1_score_per_label.append(f1)
      P_per_label.append(p)
      R_per_label.append(r)

      F1_tot += f1*weights[i]

    return F1_score_per_label, P_per_label, R_per_label, F1_tot


def Weighted_acc(y_t, y_p, weights):
    """Compute the weighted accuracy for each label
    Argument: ground truth label Y (3D), prediction (3D), class weights
    Return: array of accuracy for each label"""

    a = BinaryAccuracy() #handles the fact that y_p is not given as a binary vector
    Acc_per_label = []
    acc_tot = 0

    for i in range(8):
        a.update_state(y_t[:,i], y_p[:,i] )
        acc = a.result().numpy()
        a.reset_states()
        Acc_per_label.append(acc)
        acc_tot = acc*weights[i]
    return Acc_per_label, acc_tot

def summarize_scores(scores, names): #OK
  """Nice printing function for scores
  Arguments: scores list, names list
  Print score mean and standard deviation"""
  for score, name in zip(scores, names):
    m,s = np.mean(score), np.std(score)
    print(name, ': %.3f%% (+/-%.3f)' %(m,s))

def custom_scoring(y_true, y_pred):
  weights = np.array([0.02409584, 0.00787456, 0.03685528, 0.01760536, 0.04589969, 0.8483942 , 0.01724058, 0.00203449]);
  y_pred = y
  acc = Weighted_acc(y_true, y_pred, w)
  f1 = F1_score(y_true, y_pred, w)
  acc_tot = np.sum(acc)
  f1_tot = np.sum(f1)
  return acc, f1, acc_tot, f1_tot