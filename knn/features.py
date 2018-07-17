import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate
from sklearn.neighbors import KNeighborsClassifier


def choose_crossvalidation(stratified=True, n_splits=10, shuffle=True):
    if stratified:
        return StratifiedKFold(n_splits=n_splits, shuffle=shuffle)
    else:
        return KFold(n_splits=n_splits, shuffle=shuffle)

def compute_scores(X, y, scoring, k=5, metric='euclidean', weights='uniform', algorithm='brute', stratified=True, n_splits=10, shuffle=True):
    cv=choose_crossvalidation(stratified, n_splits, shuffle)
    clf=KNeighborsClassifier(n_neighbors=k, weights=weights, algorithm=algorithm, metric=metric)
    n_splits_scores=cross_validate(clf, X, y, scoring=scoring, cv=cv, return_train_score=False)
    scores = compute_average_metrics(n_splits_scores, ['test_'+name for name in scoring])

    return scores

def compute_average_metrics(scores, metrics):
    return {metric: (np.mean(scores[metric]), np.std(scores[metric])) for metric in metrics}  

def print_average_metrics(scores):
    for key, value in scores.items():
        avg, sd = value
        print(key + ' avg: {}, st: {}'.format(avg, sd))
  
def metrics_unpack(scores):
    accuracy, _ = scores.get('test_accuracy')
    f1score, _ = scores.get('test_f1_macro')
  
    return accuracy, f1score