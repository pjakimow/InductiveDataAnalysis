import matplotlib.pyplot as plt

import helper
import metrics_helper as metrics
import data
  
X, y = data.read_data('data', 'pima-indians-diabetes')
results = {'accuracy': [], 'precision_macro': [], 'recall_macro': [], 'f1_macro': []}

K_range = [i for i in range(3,11)]
for K in K_range:
    result = helper.run_classifier('GNB', X, y, True, K, False, 'EQUAL_STEP', 10)
    results = metrics.update_metrics_dict(results, result)
    
x_values = K_range
fig, ax = plt.subplots()
ax.plot(x_values,[item[0] for item in results['accuracy']], 'r', label='Accuracy')
ax.plot(x_values,[item[0] for item in results['precision_macro']], 'b', label='Precision')
ax.plot(x_values,[item[0] for item in results['recall_macro']], 'g', label='Recall')
ax.plot(x_values,[item[0] for item in results['f1_macro']], 'y', label='F1')
plt.ylabel('Metrics')
plt.xlabel('K - crossvalidation parameter')
legend = ax.legend(loc='lower right', shadow=True)

plt.show()