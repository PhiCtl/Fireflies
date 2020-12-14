import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import random as python_random
import joblib

from Utils import *
from Metrics import* 

from tensorflow import keras
from tensorflow.keras import optimizers, regularizers,Input, Model
from tensorflow.keras.backend import reshape, clear_session
from keras.layers import Dense, Bidirectional, Flatten, Dropout, LSTM, LeakyReLU,Activation
from keras.models import Sequential, load_model
from keras.utils import to_categorical
from focal_loss import BinaryFocalLoss
from plot_keras_history import plot_history


from sklearn import metrics
import seaborn as sn

# Import the model we are using
from sklearn.ensemble import RandomForestClassifier
from tcn import *








def run_exp_hist(x_tr,y_tr, x_te, y_te, repeats=5, gamma = 2,epochs=2, node = 600, dropout = 0.1, m_type = 1, LSTM = True, TCN = False, CV = True):
    """ Runs several experiments and averages the results
    Arguments: (x_tr, y_tr) training set
               (x_te, y_te) test set
               repeats = number of training on which we average
               gamma = parameter for the focal loss
               epochs= number of times, it passes through the training dataset.
               node = number of nodes of the neural network
               dropout = dropout value
               m_type = type of model to train/evaluate (on LSTM layers)
                        0: 1 LSTM layer model
                        1: 1 bidirectional LSTM layer model
                        2: 2 bidirectional LSTM layers model
               LSTM/TCN boolean defines which model is used.
               CV = true if cross validation (then test set is left untouched)
                    false if repeated runs (then test set is used against whole train set)
    Prints the average weighted, macro and proportional F1 score, mean F1 score, 
    precision, recall per label, and loss evolution
                  """
    #wf1_
    f1_scores = list()
    acc_scores = list()
    loss_scores = list()
    train = pd.DataFrame()
    val = pd.DataFrame()
    tab = []

    if not CV:
      print("Training versus test")
      x_1, x_2, y_1, y_2 = x_tr, x_te, y_tr, y_te

    for r in range(repeats):
        #if cross validation folds (5 folds)
        if CV:
          print("Cross validation")
          x_1, x_2, y_1, y_2 = train_test_split(x_tr, y_tr, test_size = 0.2)
        
        if LSTM:
          hist, loss, accuracy, wf1, wf1_, mf1, F1_tab, Ptab, Rtab = evaluate_model(x_1, y_1, x_2, y_2, nodes_nb = node, drop = dropout, model_type = m_type)
        
        if TCN:
             
            hist, loss, accuracy, wf1,wf1_, mf1, F1_tab, Ptab, Rtab= evaluate_model_TCN(x_1, y_1, x_2, y_2, gamma,epochs)
        
        
        wf1 = wf1 * 100.0
        mf1 = mf1 * 100.0
        wf1_ = wf1_ * 100.0
        accuracy = accuracy * 100.0
        f1_scores.append([wf1, mf1, wf1_])
        acc_scores.append(accuracy)
        loss_scores.append(loss)
        tab.append(F1_tab)
        train[str(r)] = hist.history['loss']
        val[str(r)] = hist.history['val_loss']



    f1_scores = np.array(f1_scores)
    summarize_scores([f1_scores[:,0], f1_scores[:,1], f1_scores[:,2], acc_scores, loss_scores], ['weighted F1 score', 'Macro F1 score', 'Proportional F1 score', 'Accuracy', 'Loss'])
    print("Mean F1 score per label: ", np.mean(np.array(tab), axis = 0))
    plt.plot(train, color='blue', label='train')
    plt.plot(val, color='orange', label='test')
    plt.title('model train vs test loss')
    plt.ylabel('Focal loss')
    plt.xlabel('epoch')
    plt.show()
    return f1_scores, tab


def evaluate_model(x_tr, y_tr, x_te, y_te, model_type = 1, gamma=2, nodes_nb=600, drop = 0.1, epochs = 200, verbose = 0, plot = 0, single_run = 0):
    """Training function, to evaluate train set against test set or train set againts validation set
        Arguments: (x_tr, y_tr) training data
                   (x_te, y_te) testing/validation data
                   model_type: type of model to train/evaluate (on LSTM layers)
                              0: 1 LSTM layer model
                              1: 1 bidirectional LSTM layer model
                              2: 2 bidirectional LSTM layer model
                   gamma: focal loss parameter
                   nodes_nb: number of neurons in the LSTM layers
                   drop: dropout value
                   verbose: if true, print all the metrics
                   plot: if true, print built in plot
                   single_run: fix random seed to ensure reproducibility
          Returns: loss: Last binary focal loss value on test/validation set
                  accuracy: accuracy on test/validation set 
                  wf1: weighted F1 score
                  wf1_: custom weighted F1 score (with proportional weights)
                  mf1: macro F1 score 
                  F1_tab: F1 score per label 
                  Ptab: precision per label 
                  Rtab: recall per label"""

    batch_size = 32
    n_features, n_outputs = x_tr.shape[2], y_tr.shape[2]
    w = class_weights(y_tr)
    clear_session()

    if single_run:
      np.random.seed(123)
      python_random.seed(123)
      tf.random.set_seed(1234)

    #-------------------------------------model definition-------------------------------------#
    model = Sequential()

    #model types
    if model_type == 0:
      model.add(LSTM(nodes_nb,input_shape = (None, n_features), return_sequences = True))
      model.add(Dropout(drop))
      model.add(Dense(n_outputs, activation = 'sigmoid'))
    if model_type == 1:
      model.add(Bidirectional(LSTM(nodes_nb,input_shape = (None, n_features), return_sequences = True)))#, kernel_regularizer = regularizers.l1_l2(l1=1e-6, l2=1e-5))))
      model.add(Dropout(drop))
      model.add(Dense(n_outputs, activation = 'sigmoid'))
      #model.add(Dense(n_outputs, activation = LeakyReLU(0.01)))
    if model_type == 2:
      model.add(Bidirectional(LSTM(nodes_nb,input_shape = (None, n_features), return_sequences = True)))
      model.add(Dropout(drop))
      model.add(Bidirectional(LSTM(nodes_nb,input_shape = (None, n_features), return_sequences = True)))
      model.add(Dropout(drop))
      model.add(Dense(n_outputs, activation = 'sigmoid'))
    
    model.compile(loss=BinaryFocalLoss(gamma), optimizer = 'adam', metrics = [BinaryAccuracy(), Precision(), Recall(), FalseNegatives(), FalsePositives()])
    #model.compile(loss='binary_crossentropy', optimizer = 'adam', metrics = [BinaryAccuracy(), Precision(), Recall(), FalseNegatives(), FalsePositives()], loss_weights = w)
    
    
    #------------------------------------fit network---------------------------------------------#

    hist = model.fit(x_tr, y_tr, epochs = epochs, batch_size = batch_size, verbose = 0, validation_data = (x_te, y_te))
    if verbose: 
      model.summary()

    #---------------------------------evaluate model----------------------------------------------#

    #evaluate model on test set (over all classes)
    loss, accuracy, P, R, FN, FP= model.evaluate(x_te, y_te, batch_size = batch_size, verbose = verbose)

    #save model
    if single_run:
        model.save('Results/opt_LSTM_model')
        print("Model saved to Results")

    y_pred = model.predict(x_te, batch_size = batch_size, verbose = 0)
    y_pred = reshape(y_pred, (y_pred.shape[0]* y_pred.shape[1], 8))
    y_te = reshape(y_te, (y_te.shape[0]*y_te.shape[1],8))

    #evaluate F1 score for each label
    F1_tab, Ptab, Rtab, wf1_ = F1_score(y_te, y_pred, w)
    #evaluate accuracy per label
    acc_tab = Acc(y_te, y_pred)

    #test f1 score built in
    f = F1Score(8, threshold = 0.5, average = 'weighted')
    f.update_state(y_te, y_pred)
    wf1 = f.result().numpy()
    f.reset_states()
    f = F1Score(8, threshold = 0.5, average = 'macro')
    f.update_state(y_te, y_pred)
    mf1 = f.result().numpy()
    f.reset_states()

    #edit distance
    #edit_dist_av = LevDistMultilabels(y_te, y_pred)

    #-----------------------------------------print---------------------------------------------#

    #print all
    if verbose :
      print(" -> Accuracy: ", accuracy, "; Mean of labelwise accuracy: ", np.mean(acc_tab))
      print("Per label accuracy: ", acc_tab)
      print("-> Proportional F1 score: ", wf1_, "; Weighted F1 score: ", wf1, "; Macro F1 score: ", mf1)
      print("-> F1 score per label: ", F1_tab)
      print("-> Precision: ", P, "; Recall: ", R)
      print("-> Precision per label: ", Ptab)
      print("-> Recall per label: ", Rtab)
      #print("-> Edit distance averaged: ", edit_dist_av)
      print("-> Loss: ", loss)

    if plot:
      plot_history(hist.history)

    return hist, loss, accuracy, wf1, wf1_, mf1, F1_tab, Ptab, Rtab

def evaluate_model_TCN(x_tr, y_tr, x_te, y_te, gamma=2,epochs= 5, verbose = 0,plot = 0, single_run = 0):
    """Training function, to evaluate train set against test set or train set againts validation set
        Arguments: (x_tr, y_tr) training data
                   (x_te, y_te) testing/validation data
                   gamma: focal loss parameter
                   epochs= number of times, it passes through the training dataset.
                   verbose: if true, print all the metrics
                   plot: if true, print built in plot
                   single_run: fix random seed to ensure reproducibility
                   
          Returns: loss: Last binary focal loss value on test/validation set
                  accuracy: accuracy on test/validation set 
                  wf1: weighted F1 score
                  wf1_: custom weighted F1 score (with proportional weights)
                  mf1: macro F1 score 
                  F1_tab: F1 score per label 
                  Ptab: precision per label 
                  Rtab: recall per label"""
     
    batch_size=1; 
    w = class_weights(y_tr)
    clear_session()
    
    if single_run:
      np.random.seed(123)
      python_random.seed(123)
      tf.random.set_seed(1234)
    
    #-------------------------------------model definition---------------------------#
    #Creation TCN object
    Tcn=TCN(nb_filters=64,kernel_size=2,nb_stacks=1,dilations=(8,16,32,64,128,256,512,1024),return_sequences=True,
activation=LeakyReLU(0.01),kernel_initializer='he_normal')

    i = Input(batch_shape=(1,None,  x_tr.shape[2]))
    o = Tcn(i)
    o = Dense(200, activation=LeakyReLU(0.01), kernel_regularizer=keras.regularizers.l1_l2(0.00001))(o)
    o = Dense(8, activation='sigmoid')(o)

    model = Model(inputs=[i], outputs=[o])
    model.compile(optimizer='adam', loss=BinaryFocalLoss(gamma), metrics=[BinaryAccuracy(), Precision(), Recall(), FalseNegatives(),FalsePositives()])
    
    model.summary()
    
    #---------------------fit network---------------------------------------------#
    hist = model.fit(x_tr, y_tr, epochs = epochs, batch_size = batch_size, verbose = 0, validation_data = (x_te, y_te))
    #---------------------------------evaluate model----------------------------------------------#
    
    #evaluate model on test set (over all classes)
    loss, accuracy, P, R, FN, FP= model.evaluate(x_te, y_te, batch_size = batch_size, verbose = verbose)
    
    #save model
    if single_run:
        model.save('Results/opt_TCN_model')
        print("Model saved to Results")
        
    y_pred = model.predict(x_te, batch_size = batch_size, verbose = 0)
    y_pred[y_pred<0.5]=0.
    y_pred[y_pred>0.5]=1.
    
    
    y_pred = reshape(y_pred, (y_pred.shape[0]* y_pred.shape[1], 8))
    y_te = reshape(y_te, (y_te.shape[0]*y_te.shape[1],8))

    #evaluate F1 score for each label
    F1_tab, Ptab, Rtab, wf1_ = F1_score(y_te, y_pred, w)
    
    #evaluate accuracy per label
    acc_tab = Acc(y_te, y_pred)
    print("-> F1 score per label: ", F1_tab)
    print("-> y_pred " ,y_pred[:,4])
   
    #test f1 score built in
    f = F1Score(8, threshold = 0.5, average = 'weighted')
    f.update_state(y_te, y_pred)
    wf1 = f.result().numpy()
    print("weighted F1 score built in: ", wf1 )
    f.reset_states()
    f = F1Score(8, threshold = 0.5, average = 'macro')
    f.update_state(y_te, y_pred)
    mf1 = f.result().numpy()
    print("macro F1 score built in: ", mf1 )
    f.reset_states()

    #-----------------------------------------print---------------------------------------------#

    #print all
    if verbose :
      print(" -> Accuracy: ", accuracy, "; Mean of labelwise accuracy: ", np.mean(acc_tab))
      print("Per label accuracy: ", acc_tab)
      print("-> Weighted F1 score: ", wf1_)
      print("-> F1 score per label: ", F1_tab)
      print("-> Precision: ", P, "; Recall: ", R)
      print("-> Precision per label: ", Ptab)
      print("-> Recall per label: ", Rtab)
      print("-> Loss: ", loss)

    if plot:
      plot_history(hist.history)

    return hist, loss, accuracy, wf1,wf1_, mf1, F1_tab, Ptab, Rtab


def build_model_RF(X,Y):
    
    #train and test split
    x_tr, x_te, y_tr, y_te = train_test_split(X, Y, test_size = 0.2, random_state = 200)

    #feature expansion
    x_tr_expanded=feature_expansion(x_tr)
    x_te_expanded=feature_expansion(x_te)

    #time window flattening
    x_tr_expanded,y_tr_t=time_window_sample(x_tr_expanded, y_tr, 120)
    x_te_expanded,y_te_t=time_window_sample(x_te_expanded, y_te, 120)

    #reshape into one video
    x_tr_reshaped=np.reshape(x_tr_expanded, (x_tr_expanded.shape[0]*x_tr_expanded.shape[1],x_tr_expanded.shape[2]))
    y_tr_reshaped=np.reshape(y_tr_t, (y_tr_t.shape[0]*y_tr_t.shape[1],y_tr_t.shape[2]))
    

    x_te_reshaped=np.reshape(x_te_expanded, (x_te_expanded.shape[0]*x_te_expanded.shape[1],x_te_expanded.shape[2]))
    y_te_reshaped=np.reshape(y_te_t, (y_te.shape[0]*y_te_t.shape[1],y_te_t.shape[2]))
    
    #train classifier
    clf = RandomForestClassifier(n_estimators=10, criterion = 'gini', max_depth=10, random_state = 42)
    clf.fit(x_tr_reshaped,y_tr_reshaped)
    
    #save model
    joblib.dump(clf, "./random_forest.joblib")

def predict(X,Y, flag, batch_size = 32, epochs = 200):
  """ Predicts labels for X given and compares predictions to ground truth Y
      Arguments: (X,Y) data and corresponding labels, 
                    X dim = (number of samples, number of timesteps, number of features = 75)
                    Y dim = (number of samples, number of timesteps, number of labels = 8)
                  flag: model to use
                  batch_size : should be set to 1 if only 1 sample
                  repeats: the averaged scores are computed over #number of repeats since stochastic approach
      Prints scores and downloads prediction
  """
  
  ##RESHAPE GROUND TRUTH Y
  y_te = reshape(Y, (Y.shape[0]*Y.shape[1],Y.shape[2]))

  if flag == "LSTM":
    ## LOAD MODEL
    loaded_model = load_model('Results/opt_LSTM_model')
    print("Loaded model from disk")
    loaded_model.compile(loss=BinaryFocalLoss(2), optimizer = 'adam', metrics = [BinaryAccuracy(), Precision(), Recall(), FalseNegatives(), FalsePositives()])
    #predict and reshape predictions
    y_pred = loaded_model.predict(X, batch_size = batch_size)
    y_pred = reshape(y_pred, (y_pred.shape[0]* y_pred.shape[1], 8))
    #build-in predictions  
    loss, accuracy, P, R, FN, FP = loaded_model.evaluate(X,Y, batch_size = batch_size)
  
  if flag == "TCN":
    loaded_model = load_model('Results/opt_TCN_model')
    print("Loaded model from disk")
    loaded_model.compile(optimizer='adam', loss=BinaryFocalLoss(5), metrics=[BinaryAccuracy(), Precision(), Recall(), FalseNegatives(), FalsePositives()])
    
    #predict and reshape predictions
    y_pred = loaded_model.predict(X, batch_size = batch_size)
    y_pred = reshape(y_pred, (y_pred.shape[0]* y_pred.shape[1], 8))
    #build-in predictions  
    loss, accuracy, P, R, FN, FP = loaded_model.evaluate(X,Y, batch_size = batch_size)

  if flag == "Random Forest":
    
    ##Load MODEL
    loaded_rf = joblib.load("./random_forest.joblib")
    
    #feature expansion
    X_expanded=feature_expansion(X)

    #time window flattening
    X_expanded,Y=time_window_sample(X_expanded, Y, 120)
    
    #reshape into one video
    X_reshaped=np.reshape(X_expanded, (X_expanded.shape[0]*X_expanded.shape[1],X_expanded.shape[2]))
    y_te =np.reshape(Y, (Y.shape[0]*Y.shape[1],Y.shape[2]))
    
    ##prediction
    y_pred=loaded_rf.predict(X_reshaped)


  #customed predictions
  wf1, mf1, pf1, F1_tab, Ptab, Rtab, acc_tab = custom_scoring(y_te, y_pred)

  print("F1 score per label: ", F1_tab)
  print("Precision per label: ", Ptab)
  print("Recall per label: ", Rtab)
  print("Macro F1 score: ", mf1, " ; Weighted F1 score: ", wf1, " ; Proportional F1 score: ", pf1)
  
  y_pred = y_pred > 0.5
  print("Prediction will be saved into Results/")
  name = 'Results/'+flag+'_Annotation.csv'
  pd.DataFrame(y_pred, columns=["arch", "burrow + arch", "drag + arch", "groom","tap + arch","egg", "proboscis", "idle"]).to_csv(name)