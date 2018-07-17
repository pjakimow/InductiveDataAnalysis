import numpy as np

def print_average_metrics(scores):
    for key, value in scores.items():
         avg, sd = value
         print(key + ' avg: {}, st: {}'.format(avg, sd))
    
def compute_average_metrics(scores, metrics):
    return {metric: (np.mean(scores[metric]), np.std(scores[metric])) for metric in metrics}  
    
def update_metrics_dict(scores, score):  
    for metric in scores.keys():
      scores[metric]+=[score['test_' + metric]]
    return scores