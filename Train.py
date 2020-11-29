import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from keras.models import Sequential
from tensorflow.keras import optimizers
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


def run_exp_hist(x_1, y_1, x_2, y_2, repeats=5, gamma = 2, node = 100):
    f1_scores = list()
    acc_scores = list()
    train = pd.DataFrame()
    val = pd.DataFrame()

    for r in range(repeats):
        acc, f1, hist = evaluate_model(x_1, y_1, x_2, y_2, gamma, node)
        f1 = f1 * 100.0
        acc = acc * 100.0
        print('>#%d: %.3f %.3a' %(r, f1, acc))
        f1_scores.append(f1)
        acc_scores.append(acc)
        train[str(r)] = hist.history['loss']
        val[str(r)] = hist.history['val_loss']
    summarize_scores([f1_scores, acc_scores], ['F1 score', 'Accuracy'])

    plt.plot(train, color='blue', label='train')
    plt.plot(val, color='orange', label='test')
    plt.title('model train vs test loss')
    plt.ylabel('Focal loss')
    plt.xlabel('epoch')
    plt.show()
    
def evaluate_model(x_tr, y_tr, x_te, y_te, gamma=2, nodes_nb=100):
    verbose, epochs, batch_size = 0, 100, 20
    n_features, n_outputs = x_tr.shape[2], y_tr.shape[2]
    _,_, w = class_weights(y_tr)
    
    #-------------------------------------model definition---------------------------#
    model = Sequential()
    model.add(LSTM(nodes_nb,input_shape = (None, n_features), return_sequences = True))
    #model.add(Dropout(0.5))
    #model.add(Dense(nodes_nb, activation='relu'))
    model.add(Dense(n_outputs, activation = 'sigmoid'))
    model.compile(loss=BinaryFocalLoss(gamma), optimizer = 'adam', metrics = [BinaryAccuracy(), Precision(), Recall(), FalseNegatives(), FalsePositives()])
    #model.compile(loss='binary_crossentropy', optimizer = 'adam', metrics = [BinaryAccuracy(), Precision(), Recall(), FalseNegatives(), FalsePositives()], loss_weights = w)
    
    model.summary()
    
    #---------------------fit network---------------------------------------------#

    hist = model.fit(x_tr, y_tr, epochs = epochs, batch_size = batch_size, verbose = verbose, validation_data = (x_te, y_te))
    #model.save_weights("model.h5")
    #print("Saved model to disk")

    #evaluate model on test set (over all classes)
    loss, accuracy, P, R, FN, FP= model.evaluate(x_te, y_te, batch_size = batch_size, verbose = verbose)

    y_pred = model.predict(x_te, batch_size = batch_size, verbose = verbose)
    y_pred = reshape(y_pred, (y_pred.shape[0]* y_pred.shape[1], 8))
    y_te = reshape(y_te, (y_te.shape[0]*y_te.shape[1],8))

    #evaluate F1 score for each label
    F1_tab, wf1 = F1_score(y_te, y_pred, w)
    #evaluate accuracy per label
    acc_tab, wacc = Weighted_acc(y_te, y_pred, w)

    #test f1 score built in
    f = F1Score(8, threshold = 0.5, average = 'weighted')
    f.update_state(y_te, y_pred)
    print("weighted F1 score built in: ", f.result().numpy() )
    f.reset_states()
    f = F1Score(8, threshold = 0.5, average = 'macro')
    f.update_state(y_te, y_pred)
    print("macro F1 score built in: ", f.result().numpy() )
    f.reset_states()

    #test accuracy
    a = BinaryAccuracy()
    a.update_state(y_te, y_pred)
    print("Final acc: ", a.result().numpy())
    a.reset_states()

    #print all
    print(" -> Accuracy: ", accuracy, "; Mean of labelwise accuracy: ", np.mean(acc_tab), "Weighted accuracy:", wacc)
    print("Per label accuracy: ", acc_tab)
    print("-> Weighted F1 score: ", wf1)
    print("-> F1 score per label: ", F1_tab)
    print("-> Precision: ", P, "; Recall: ", R)
    print("-> Loss: ", loss)
    return hist, y_pred, y_te

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



def cross_validation(x_tr, y_tr, x_val, y_val, nodes, gamma = 2):
    scores_f1 = list()
    accuracy = list()
    train = pd.DataFrame()
    val = pd.DataFrame()
    colours = ['blue','green','orange','red','yellow','pink','black','grey']
    for i, n in enumerate (nodes):
        a, f1, hist = evaluate_model(x_tr, y_tr, x_val, y_val, gamma, n)
        f1 = f1 * 100.0
        a = a*100.0
        print('>#%i: f1 score: %.3f1, accuracy: %.5a' %(i, f1, a))
        scores_f1.append(f1)
        accuracy.append(a)
        train[str(i)] = hist.history['loss']
        val[str(i)] = hist.history['val_loss']
        plt.plot(train[str(i)], color=colours[2*i], label = i)
        plt.plot(val[str(i)], color=colours[2*i+1], label = i)

    summarize_scores(scores_f1, 'F1_score')
    summarize_scores(accuracy, 'Accuracy')
    plt.title('model train vs validation loss')
    plt.ylabel('Focal loss')
    plt.xlabel('epoch')
    plt.legend()
    plt.show()
