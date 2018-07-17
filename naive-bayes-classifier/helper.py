from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate

import discretization as discr
import metrics_helper as metrics

def choose_discretization(X, mode, bins_count):
    if mode == "EQUAL_STEP":
        return discr.constant_range_step(X, bins_count)
    elif mode == "EQUAL_QUANTITY":
        return discr.equal_elements_number(X, bins_count)
    elif mode == "LOG_STEP":
        return discr.log_step(X, bins_count)
    return X

def choose_crossvalidation(stratified, n_splits, shuffle):
    if stratified:
        return StratifiedKFold(n_splits=n_splits, shuffle=shuffle)
    else:
        return KFold(n_splits=n_splits, shuffle=shuffle)

def choose_model(mode):
    if mode == 'MNB':
        return MultinomialNB()
    elif mode == 'GNB()':
        return GaussianNB()
    return GaussianNB()

def run_classifier(mode, X, y, stratified, K, crossval_shuffle, discr_mode, bins_quantity, scoring={'accuracy', 'precision_macro', 'recall_macro', 'f1_macro'}):
    X = choose_discretization(X, discr_mode, bins_quantity)
    kf = choose_crossvalidation(stratified, K, crossval_shuffle)
    nb = choose_model(mode)
    results = cross_validate(nb, X, y, scoring=scoring, cv=kf, return_train_score=False)
    scores = metrics.compute_average_metrics(results, ['test_'+name for name in scoring])
  
    return scores