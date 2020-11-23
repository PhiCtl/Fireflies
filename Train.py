import numpy as np

#need to write evaluate_model here:
def evaluate_model:
    raise NotImplementedError

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

