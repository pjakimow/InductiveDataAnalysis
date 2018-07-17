import matplotlib.pyplot as plt
import features as f

def test_metrics_and_k(X, y, scoring, k_test_set, metrics, weight, stratified, n_splits):
    #plot settings
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_prop_cycle('color', ['tab:pink','tab:purple','tab:green','tab:olive'])

    #compute result
    for metric in metrics:
        accuracies, f1scores = [], []
        
        for k in k_test_set:
            scores = f.compute_scores(X, y, scoring, k=k, metric=metric, weights=weight, algorithm='brute', stratified=stratified, n_splits=n_splits, shuffle=True)
            acc, f1 = f.metrics_unpack(scores)  
            accuracies += [acc]
            f1scores += [f1]
         
        print('acc, distance={}'.format(metric))
        print([ '%.3f' % elem for elem in accuracies])
        print('f1, distance={}'.format(metric))
        print([ '%.3f' % elem for elem in f1scores])
        
        #plot
        acc_label = 'accuracy, distance=%s' %(metric)
        f1_label = 'f1-score, distance=%s' %(metric) 
        line1, = plt.plot(k_test_set, accuracies, '--', label=acc_label, lw=4)
        line2, = plt.plot(k_test_set, f1scores, label=f1_label, lw=4)
        plt.legend()
        
    plt.xlabel('k - knn parameter')
    plt.ylabel('metrics')
   
    plt.show()

def test_weights_and_k(X, y, scoring, k_test_set, weights, metric, stratified, n_splits):
    #plot settings
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_prop_cycle('color', ['tab:pink','tab:purple','tab:green','tab:olive'])
    
    #compute result
    for weight in weights:
        accuracies, f1scores = [], []
        
        for k in k_test_set:
            scores = f.compute_scores(X, y, scoring, k=k, metric=metric, weights=weight, algorithm='brute', stratified=stratified, n_splits=n_splits, shuffle=True)
            acc, f1 = f.metrics_unpack(scores)  
            accuracies += [acc]
            f1scores += [f1]
        
        print('acc, weight={}'.format(weight))
        print([ '%.3f' % elem for elem in accuracies])
        print('f1, weight={}'.format(weight))
        print([ '%.3f' % elem for elem in f1scores])
        
        #plot
        acc_label = 'accuracy, weight=%s' %(weight)
        f1_label = 'f1-score, weight=%s' %(weight)    
        line1, = plt.plot(k_test_set, accuracies, '--', label=acc_label, lw=4)
        line2, = plt.plot(k_test_set, f1scores, label=f1_label, lw=4)
        plt.legend()
        
    plt.xlabel('k - knn parameter')
    plt.ylabel('metrics')  
    plt.show()

def test_ncrossvalidation_with_const_k(X, y, n_test_set, scoring, k, metric, weight):
    #plot settings
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_prop_cycle('color', ['tab:pink','tab:purple','tab:green','tab:olive'])
    
    #compute
    for stratified in [True, False]:
        accuracies, f1scores = [], []
        for n in n_test_set:
            scores = f.compute_scores(X, y, scoring, k=k, metric=metric, weights=weight, algorithm='brute', stratified=stratified, n_splits=n, shuffle=True)
            acc, f1 = f.metrics_unpack(scores)  
            accuracies += [acc]
            f1scores += [f1]
        
        print('acc, stratified={}'.format(stratified))
        print([ '%.3f' % elem for elem in accuracies])
        print('f1, stratified={}'.format(stratified))
        print([ '%.3f' % elem for elem in f1scores])
        
        #plot    
        acc_label = 'accuracy, stratified=%s' %(stratified)
        f1_label = 'f1-score, stratified=%s' %(stratified)    
        line1, = plt.plot(n_test_set, accuracies, '--', label=acc_label, lw=4)
        line2, = plt.plot(n_test_set, f1scores, label=f1_label, lw=4)
        plt.legend()
    
    plt.xlabel('k - crossvalidation parameter')
    plt.ylabel('metrics')
    plt.show()
    
def test_ncrossvalidation_with_const_n(X, y, k_test_set, scoring, n, metric, weight):
    #plot settings
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_prop_cycle('color', ['tab:pink','tab:purple','tab:green','tab:olive'])
    
    #compute
    for stratified in [True, False]:
        accuracies, f1scores = [], []
        for k in k_test_set:
            scores = f.compute_scores(X, y, scoring, k=k, metric=metric, weights=weight, algorithm='brute', stratified=stratified, n_splits=n, shuffle=True)
            acc, f1 = f.metrics_unpack(scores)  
            accuracies += [acc]
            f1scores += [f1]
        
        print('acc, stratified={}'.format(stratified))
        print([ '%.3f' % elem for elem in accuracies])
        print('f1, stratified={}'.format(stratified))
        print([ '%.3f' % elem for elem in f1scores])
        
        #plot    
        acc_label = 'accuracy, stratified=%s' %(stratified)
        f1_label = 'f1-score, stratified=%s' %(stratified)    
        line1, = plt.plot(k_test_set, accuracies, '--', label=acc_label, lw=4)
        line2, = plt.plot(k_test_set, f1scores, label=f1_label, lw=4)
        plt.legend()
    
    plt.xlabel('k - knn parameter')
    plt.ylabel('metrics')
    plt.show()
