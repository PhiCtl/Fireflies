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

def LevenshteinDistance(a, b):
    """From Wagner-Fischer algorithm (source: wikipedia)
       Computes edit distance between two strings of length n and m
       Min number of operations (insertions, deletions, substitutions)
       To turn a into b
    Arguments: a is the source of length m
               b is the target of length n
    Returns :  d matrix of edit-distance"""
    m = len(a)
    n = len(b)
    d = np.zeros((m,n))
    
    #if target is empty, source can be turned into target by dropping character
    for i in np.arange(1,m):
        d[i,0] = i
    #if source is empty, source can be turned into target by inserting character
    for j in np.arange(1,n):
        d[0,j] = j
        
    for j in np.arange(0, n):
        for i in np.arange(0, m):
            substCost = 0
            if (a[i] != b[j]):
                substCost = 1
            #the min distance between two strings of length i and j is the
            #min between deleting, inserting or subsituting
            d[i,j] = min(d[i-1,j] + 1, d[i,j-1] + 1, d[i-1, j-1] + substCost)
    
    return d

def LevDistMultilabels(y_true, y_pred):
    """Computes edit distance between y_pred and y_true
    Arguments: 2D y_true of same dimensions as below
               2D y_pred of dim(number of time steps, number of categories)
    Returns: mean edit distance"""
    
    n = y_pred.shape[0]
    D = 0
    for i in range(n):
        D += LevenshteinDistance(y_pred[i,:], y_true[i,:])[-1, -1]
    return D/n

def custom_scoring(y_te, y_pred):
  """Custom scores for predict function
     Arguments: y_true :ground truth label vector (2D flattened)
                y_pred: predicted labels vector (2D flattened)
     Returns: weighted f1 score
              macro f1 score
              proportional f1 score
              F1 score per label
              precision per label
              recall per label
              accuracy per label
              edit distance average """
  #weights computed with training data set
  w = np.array([0.02409584, 0.00787456, 0.03685528, 0.01760536, 0.04589969, 0.8483942 , 0.01724058, 0.00203449]);
  
  ## F1 SCORES
  #evaluate F1 score, precision and recall for each label, 
  #along with custom proportionally weighted F1 score
  #and built in weighted and macro F1 scores
  F1_tab, Ptab, Rtab, pf1 = F1_score(y_te, y_pred, w)
  f = F1Score(8, threshold = 0.5, average = 'weighted')
  f.update_state(y_te, y_pred)
  wf1 = f.result().numpy() #weighted f1 score
  f.reset_states()
  f = F1Score(8, threshold = 0.5, average = 'macro')
  f.update_state(y_te, y_pred)
  mf1 = f.result().numpy() #macro f1 score
  f.reset_states()

  ##EDIT DISTANCE
  #edit_dist_av = LevDistMultilabels(y_true, y_pred)

  ##ACCURACY
  #evaluate accuracy per label
  acc_tab = Acc(y_te, y_pred)

  return wf1, mf1, pf1, F1_tab, Ptab, Rtab, acc_tab


def perf_measure(y_te, y_pred):
    TP = np.zeros(8)
    FN = np.zeros(8)
    FP = np.zeros(8)
    TN = np.zeros(8)
    for i in range(y_pred.shape[1]):
      tp = TruePositives()
      fn = FalseNegatives()
      fp = FalsePositives()
      tn = TrueNegatives()
      tp.update_state(y_te[:,i], y_pred[:,i])
      fn.update_state(y_te[:,i], y_pred[:,i])
      fp.update_state(y_te[:,i], y_pred[:,i])
      tn.update_state(y_te[:,i], y_pred[:,i])
      TP[i] = tp.result().numpy()
      FN[i] = fn.result().numpy()
      FP[i] = fp.result().numpy()
      TN[i] = tn.result().numpy()
      tp.reset_states()
      fn.reset_states()
      fp.reset_states()
      tn.reset_states()
    return [TP, TN, FN, FP]
