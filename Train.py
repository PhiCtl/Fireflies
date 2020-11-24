import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from keras.models import Sequential
from tensorflow.keras import optimizers
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import LSTM
from keras.utils import to_categorical
from focal_loss import BinaryFocalLoss
from tensorflow.keras.metrics import Precision, Recall, TruePositives, FalsePositives, FalseNegatives, Accuracy
from sklearn.metrics import f1_score
from plot_keras_history import plot_history
from Utils import class_weights, binary_CE_weighted

#need to write evaluate_model here:
"""def evaluate_model:
    raise NotImplementedError"""

def summarize_scores(scores,score_name):
    """Nice printing function for 1 score only...
    Argument: scores vector, score_name
    Void return: prints confidence interval for this score"""
    print(scores)
    m,s = np.mean(scores), np.std(scores)
    print(score_name, ': %.3f%% (+/-%.3f)' %(m,s))


def run_exp(x_tr, y_tr, x_val, y_val, repeats=3, gamma = 1, node = 100 ):
    """Average prediction scores over several models
    Argument: train and validation sets, number of repeats, gamma parameter for focal loss, number of node in network
    Void return: Print metrics summary
    NOT READY YET: NEEDS TO BE MODIFIED IF YOU WANT TO PRINT SEVERAL SCORES AND RUN OVER DIFFERENT HYPERPARAMETERS"""
    scores = list()
    for r in range(repeats):
        score1 = evaluate_model(x_tr, y_tr, x_val, y_val, gamma, node)
        score1 = score1 * 100.0
        print('>#%d: %.3f' %(1, score1))
        scores.append(score1)
    summarize_scores(scores)
    
def F1_score(y_true, y_pred, weights):
    """Compute the weighted F1_score for each label
    Argument: ground truth label Y (3D), prediction (3D), class weights
    Return: array of F1_score for each label"""
    m = Precision()
    n = Recall()
    F1_score_per_label = []
    for i in range(8):
        m.update_state(y_true[:,:,i], y_pred[:,:,i])
        n.update_state(y_true[:,:,i], y_pred[:,:,i])
        p = n.result().numpy()
        r = m.result().numpy()
        if p + r == 0:
            F1_score = 0
        else:
            F1_score = 2* p * r / (p + r) #F1 score computed as harminic mean of precision and recall
            F1_score_per_label.append(F1_score*weights[i])
    return F1_score_per_label
    
def Acc(y_true, y_pred, weights):
    """Compute the weighted accuracy for each label
    Argument: ground truth label Y (3D), prediction (3D), class weights
    Return: array of accuracy for each label"""
    a = Accuracy()
    Acc_per_label = []
    for i in range(8):
        a.update_state(y_true[:,:,i], y_pred[:,:,i])
        acc = a.result().numpy()
        Acc_per_label.append(acc*weights[i])
    return Acc_per_label

def run_exp_hist(x_1, y_1, x_2, y_2, repeats=5, gamma = 2, node = 100):
    scores = list()
    train = pd.DataFrame()
    val = pd.DataFrame()
    for r in range(repeats):
        acc, score1, hist = evaluate_model(x_1, y_1, x_2, y_2, gamma, node)
        score1 = score1 * 100.0
        print('>#%d: %.3f' %(1, score1))
        scores.append(score1)
        train[str(r)] = hist.history['loss']
        val[str(r)] = hist.history['val_loss']
    summarize_scores(scores, 'F1 score')
    plt.plot(train, color='blue', label='train')
    plt.plot(val, color='orange', label='validation')
    plt.title('model train vs validation loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.show()
    
def evaluate_model(x_tr, y_tr, x_te, y_te, gamma=2, nodes_nb=100):
    verbose, epochs, batch_size = 0, 500, 64
    n_timesteps, n_features, n_outputs = x_tr.shape[1], x_tr.shape[2], y_tr.shape[2]
    pos,neg, w = class_weights(y_tr)
    
    #model definition
    model = Sequential()
    model.add(LSTM(nodes_nb,input_shape = (n_timesteps, n_features), return_sequences = True))
    model.add(Dropout(0.5))
    #model.add(LSTM(nodes_nb,input_shape = (n_timesteps, n_features), return_sequences = True))
    model.add(Dropout(0.5))
    model.add(Dense(nodes_nb, activation='relu'))
    model.add(Dense(n_outputs, activation = 'sigmoid'))

    model.compile(loss=BinaryFocalLoss(gamma), optimizer = 'adam', metrics = ['accuracy',Precision(), Recall(), FalseNegatives(), FalsePositives()]) #activity recognition metrics
    #model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy', Precision(), Recall(), FalseNegatives(), FalsePositives()], loss_weights = w)
    #doest work model.compile(loss = binary_CE_weighted(pos, neg), optimizer = 'adam', metrics = ['accuracy', Precision(), Recall(), FalseNegatives(), FalsePositives()])
    model.summary()
    
    #fit network
    hist = model.fit(x_tr, y_tr, epochs = epochs, batch_size = batch_size, verbose = verbose, validation_data = (x_te, y_te))
    old_weights = model.get_weights()
    new_model.set_weights(old_weights)
    y_pred = model.predict(x_te, batch_size = batch_size, verbose = verbose)
    #evaluate model on test set (over all classes)
    _, accuracy, P, R, FN, FP= model.evaluate(x_te, y_te, batch_size = batch_size, verbose = verbose)
    
    #evaluate F1 score for each label
    F1_tab = F1_score(y_te, y_pred, w)
    
    #print all
    print(" -> Accuracy: ", accuracy, "; False positives: ", FP, "; False Negatives: ", FN)
    print("-> weighted F1 score: ", np.sum(F1_tab))
    print("F1 score per label: ", F1_tab)
    print("-> Precision: ", P, "; Recall: ", R)
    return accuracy, np.sum(F1_tab), hist #metrics evaluation on validation / test set
    
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
