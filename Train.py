import numpy as np
from tensorflow.keras.metrics import Precision, Recall, Accuracy

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
    
def F1_score(y_true, y_pred, class_weights):
    """Compute the F1_score for each label, weighted with its absolute proportion
    Argument: ground truth label Y (3D), prediction (3D), class_weights (inverse of proportion)
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
            F1_score_per_label.append(F1_score*class_weights[i])
    return F1_score_per_label
    
def Acc(y_true, y_pred, class_weights):
    """Compute the accuracy for each label, weighted with its absolute proportion
    Argument: ground truth label Y (3D), prediction (3D), class_weights (inverse of proportion)
    Return: array of accuracy for each label"""
    a = Accuracy()
    Acc_per_label = []
    for i in range(8):
        a.update_state(y_true[:,:,i], y_pred[:,:,i])
        acc = a.result().numpy()
        Acc_per_label.append(acc*class_weights[i])
    return Acc_per_label
    
    

