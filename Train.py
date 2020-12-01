import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from keras.models import Sequential
from tensorflow.keras import optimizers, regularizers
from tensorflow.keras.backend import reshape
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import LSTM
from keras.utils import to_categorical
from focal_loss import BinaryFocalLoss
from plot_keras_history import plot_history

from Utils import class_weights
from Metrics import* 


def run_exp_hist(x_1, y_1, x_2, y_2, repeats=5, gamma = 2, node = 300):
    f1_scores = list()
    acc_scores = list()
    loss_scores = list()
    train = pd.DataFrame()
    val = pd.DataFrame()
    tab = np.zeros((1,8))
    for r in range(repeats):
        hist, loss, accuracy, wf1, mf1, F1_tab, Ptab, Rtab = evaluate_model(x_1, y_1, x_2, y_2, gamma, node)
        wf1 = wf1 * 100.0
        mf1 = mf1 * 100.0
        accuracy = accuracy * 100.0
        print('Repeat, wf1, mf1, accuracy')
        print('>#%d: %.3f %.3g %.5a' %(r, wf1, mf1, accuracy))
        f1_scores.append([wf1, mf1])
        acc_scores.append(accuracy)
        loss_scores.append(loss)
        tab += F1_tab
        print("F1 score per label: ", F1_tab)
        train[str(r)] = hist.history['loss']
        val[str(r)] = hist.history['val_loss']
    f1_scores = np.array(f1_scores)
    summarize_scores([f1_scores[:,0], f1_scores[:,1], acc_scores, loss_scores], ['weighted F1 score', 'Macro F1 score', 'Accuracy', 'Loss'])
    print("Mean F1 score per label: ", tab/repeats)
    print("Mean precision per label: ", Ptab/repeats)
    print("Mean recall per label: ", Rtab/repeats)
    plt.plot(train, color='blue', label='train')
    plt.plot(val, color='orange', label='test')
    plt.title('model train vs test loss')
    plt.ylabel('Focal loss')
    plt.xlabel('epoch')
    plt.show()
    
def evaluate_model(x_tr, y_tr, x_te, y_te, gamma=2, nodes_nb=100, drop = 0.2, verbose = 0):
    epochs, batch_size = 200, 20
    n_features, n_outputs = x_tr.shape[2], y_tr.shape[2]
    _,_, w = class_weights(y_tr)
    
    #-------------------------------------model definition---------------------------#
    model = Sequential()
    model.add(LSTM(nodes_nb,input_shape = (None, n_features), return_sequences = True, kernel_regularizer = regularizers.l1_l2(l1=1e-5, l2=1e-3)))
    model.add(Dropout(drop))
    #model.add(Dense(nodes_nb, activation='relu'))
    model.add(Dense(n_outputs, activation = 'sigmoid'))
    model.compile(loss=BinaryFocalLoss(gamma), optimizer = 'adam', metrics = [BinaryAccuracy(), Precision(), Recall(), FalseNegatives(), FalsePositives()])
    #model.compile(loss='binary_crossentropy', optimizer = 'adam', metrics = [BinaryAccuracy(), Precision(), Recall(), FalseNegatives(), FalsePositives()], loss_weights = w)
    
    if verbose: 
      model.summary()
    
    #---------------------fit network---------------------------------------------#

    hist = model.fit(x_tr, y_tr, epochs = epochs, batch_size = batch_size, verbose = 0, validation_data = (x_te, y_te))
    #model.save_weights("model.h5")
    #print("Saved model to disk")

    #evaluate model on test set (over all classes)
    loss, accuracy, P, R, FN, FP= model.evaluate(x_te, y_te, batch_size = batch_size, verbose = verbose)

    y_pred = model.predict(x_te, batch_size = batch_size, verbose = 0)
    y_pred = reshape(y_pred, (y_pred.shape[0]* y_pred.shape[1], 8))
    y_te = reshape(y_te, (y_te.shape[0]*y_te.shape[1],8))

    #evaluate F1 score for each label
    F1_tab, Ptab, Rtab, wf1_ = F1_score(y_te, y_pred, w)
    #evaluate accuracy per label
    acc_tab, wacc = Weighted_acc(y_te, y_pred, w)

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

    #test accuracy
    a = BinaryAccuracy()
    a.update_state(y_te, y_pred)
    print("Final accuracy: ", a.result().numpy())
    a.reset_states()

    #print all
    if verbose :
      print(" -> Accuracy: ", accuracy, "; Mean of labelwise accuracy: ", np.mean(acc_tab), "Weighted accuracy:", wacc)
      print("Per label accuracy: ", acc_tab)
      print("-> Weighted F1 score: ", wf1_)
      print("-> F1 score per label: ", F1_tab)
      print("-> Precision: ", P, "; Recall: ", R)
      print("-> Precision per label: ", Ptab)
      print("-> Recall per label: ", Rtab)
      print("-> Loss: ", loss)
    return hist, loss, accuracy, wf1, mf1, F1_tab, Ptab, Rtab

"""def predict(X,Y, batch_size = 20, repeats = 6):
  loaded_model.load_weights("model.h5")
  print("Loaded model from disk")

  scores = list()
  for r in range(repeats):
    loaded_model.compile(loss=BinaryFocalLoss, optimizer='adam', metrics=['accuracy',Precision(), Recall(), FalseNegatives(), FalsePositives()])
    y_pred = model.predict(X, batch_size = batch_size)
    #customed predictions
    acc, f1, acc_tot, f1_tot = custom_scoring(Y, y_pred)
    #build-in predictions
    loss, unweighted_acc, P, R, FP, FN = model.evaluate(X,Y, batch_size = batch_size)
    scores.append([acc,f1,acc_tot,f1_tot, unweighted_acc, P, R])
  summarize(np.array(scores))
  print("Prediction will be saved into Predictions/")
  pd.DataFrame(y_pred, columns=["arch", "burrow + arch", "drag + arch", "groom","tap + arch","egg", "proboscis", "idle"]).to_csv('Predictions/Annotation.csv')
"""

def cross_validation(x_tr, y_tr, x_te, y_te, nodes, gamma = 2):
    f1_scores = list()
    acc_scores = list()
    loss_scores = list()

    for n in nodes:
        hist, loss, accuracy, wf1, mf1, F1_tab, Ptab, Rtab = evaluate_model(x_tr, y_tr, x_te, y_te, nodes_nb = n)
        wf1 = wf1 * 100.0
        mf1 = mf1 * 100.0
        accuracy = accuracy * 100.0
        print('Node, wf1, mf1, accuracy')
        print('>%d: %.3f %.5g %.5a' %(n, wf1, mf1, accuracy))
        print('Loss:', loss)
        f1_scores.append([wf1, mf1])
        acc_scores.append(accuracy)
        loss_scores.append(loss)
        print("F1 score per label: ", F1_tab)
        print("Precision per label: ", Ptab)
        print("Recall per label: ", Rtab)
    f1_scores = np.array(f1_scores)
    loss_scores = np.array(loss_scores)
    opt_drop_wf1 = nodes[np.argmax(f1_scores[:,0])]
    opt_drop_mf1 = nodes[np.argmax(f1_scores[:,1])]
    opt_drop_loss = nodes[np.argmin(loss_scores)] 

    print("Optimal node with respect to macro F1 score : ", opt_drop_wf1)
    print("Optimal node with respect to weighted F1 score : ", opt_drop_mf1)   
    print("Optimal node with respect to loss : ", opt_drop_loss)    

