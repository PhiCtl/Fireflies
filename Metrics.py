import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.metrics import Precision, Recall, TruePositives, TrueNegatives, FalsePositives, FalseNegatives, Accuracy, BinaryAccuracy
from tensorflow_addons.metrics import F1Score, MultiLabelConfusionMatrix

def F1_score(y_t, y_p, weights):
    """Computes the weighted F1_score for each label
    Argument: ground truth label Y (3D flattened into 2D), prediction (3D flattened into 2D), class weights
    Returns: array of F1_score for each label, and weighted F1 score"""

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

def Acc(y_t, y_p):
    """Computes accuracy for each label
    Argument: ground truth label Y (2D reshaped), prediction (2D reshaped)
    Returns: array of accuracy for each label"""

    a = BinaryAccuracy() #handles the fact that y_p is not given as a binary vector
    Acc_per_label = []

    for i in range(8):
        a.update_state(y_t[:,i], y_p[:,i] )
        acc = a.result().numpy()
        a.reset_states()
        Acc_per_label.append(acc)
    return Acc_per_label

def summarize_scores(scores, names): 
  """Nice printing function for scores
  Arguments: scores list, names list
  Prints score mean and standard deviation"""
  for score, name in zip(scores, names):
    m,s = np.mean(score), np.std(score)
    print(name, ': %.3f%% (+/-%.3f)' %(m,s))

def custom_scoring(y_true, y_pred):
  """Custom scores for predict function
     Arguments: y_true :ground truth label vector (2D flattened)
                y_pred: predicted labels vector (2D flattened)
     Returns: weighted f1 score
              macro f1 score
              F1 score per label
              precision per label
              recall per label
              accuracy per label """
  w = np.array([0.02409584, 0.00787456, 0.03685528, 0.01760536, 0.04589969, 0.8483942 , 0.01724058, 0.00203449]);
  
  ## F1 SCORES
  #evaluate F1 score, precision and recall for each label, 
  #along with custom proportionally weighted F1 score
  #and built in weighted and macro F1 scores
  F1_tab, Ptab, Rtab, _ = F1_score(y_te, y_pred, w)
  f = F1Score(8, threshold = 0.5, average = 'weighted')
  f.update_state(y_te, y_pred)
  wf1 = f.result().numpy() #weighted f1 score
  f.reset_states()
  f = F1Score(8, threshold = 0.5, average = 'macro')
  f.update_state(y_te, y_pred)
  mf1 = f.result().numpy() #macro f1 score
  f.reset_states()

  ##ACCURACY
  #evaluate accuracy per label
  acc_tab = Acc(y_te, y_pred, w)

  return wf1, mf1, F1_tab, Ptab, Rtab, acc_tab