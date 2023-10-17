#%%
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from utils.elpv_reader import ElpvData
from utils.sift import Sift
from utils.pre_processing import bovw
from utils.models import grid_search


#%%

N_FEATURES = 100

# Step 1: Get Descriptor
elvp = ElpvData()
sift = Sift(images=elvp.path, 
            nfeatures=N_FEATURES, 
            contrastThreshold=0.01)

descriptors = sift.descriptor
print(len(descriptors))

if sift.zero_des:
    print('Empty descriptor in images: ', sift.zero_des)


    
# Split to train, test, val
#%%
X_train, X_test, y_train, y_test = train_test_split(descriptors, 
                                                    elvp.label, 
                                                    test_size=0.2, 
                                                    random_state=99,
                                                    stratify=elvp.label)

print(len(X_train), len(X_test))
#%%

# Obtain clusters,
# KS -> voca size
ks = [20, 24, 28, 32, 36, 64, 128, 256, 512]
grid_result = sift.calculate_clusters(X_train, ks)

# %%
knn_params = {
    'n_neighbors': [5,7,9,11,13, 15, 17, 19, 21],
}
svm_params = {
    'C': [0.01, 0.1, 1, 10, 100],
    'kernel': ('linear', 'poly', 'rbf', 'sigmoid'),
    
}

forest_params = {
    'n_estimators': [50, 100, 150, 200],
    'criterion': ('gini', 'entropy'),
    'max_depth':[None, 5, 10]
}
knn = KNeighborsClassifier(metric='euclidean')
svm = SVC(tol=1e-3)
forst = RandomForestClassifier()

params = [knn_params, svm_params, forest_params]
clfs = [knn, svm, forst]

results = []

#%%

for k in ks:
    # Put clusters into bins, count number of descriptors in each bin
    hist_train = bovw(grid_result[k], X_train, method='SK')
    # hist_val = bovw(grid_result[k], X_val, method='SK')
    hist_test = bovw(grid_result[k], X_test, method='SK')
    # train a classifier use histogram

    for clf, param in zip(clfs, params):
    
        clf, pred, acc = grid_search(hist_train, y_train,
                                                hist_test, y_test,
                                                clf, param, 
                                                cv = 5, scoring='accuracy')
        

        results.append({'k': k, 
                        'clf': clf, 
                        'params': clf.get_params(), 
                        'pred': pred,
                        'acc': acc})

"""
 'kneighbors',
 All neighbors in the training set, can be used for visualisation
 
 'kneighbors_graph',
 <2099x2099 sparse matrix of type '<class 'numpy.float64'>'
	with 10495 stored elements in Compressed Sparse Row format>
"""

the_best = max(results, key=lambda x: x['acc'])
print("------"*50)
print('Best K:', the_best['k'])
print('Average acc:', the_best['acc'])
print('Parameters:', the_best['params'])

# First result
# Best K: 128
# Average acc: 0.72
# Parameters: {'C': 10, 
#   'break_ties': False, 
#   'cache_size': 200, 
#   'class_weight': None, 
#   'coef0': 0.0, 
#   'decision_function_shape': 'ovr', 
#   'degree': 3, 
#   'gamma': 'scale', 
#   'kernel': 'rbf', 
#   'max_iter': -1, 
#   'probability': False, 
#   'random_state': None, 
#   'shrinking': True, 
#   'tol': 0.001, 
#   'verbose': False}

# Report:
# Classification report:
#               precision    recall  f1-score   support

#            0       0.75      0.88      0.81       302
#            1       0.55      0.27      0.36        59
#            2       0.67      0.10      0.17        21
#            3       0.67      0.65      0.66       143

#     accuracy                           0.72       525
#    macro avg       0.66      0.48      0.50       525
# weighted avg       0.70      0.72      0.70       525

# Class 0: 302 out of 525: 57.52%
# Class 1: 59 out of 525: 11.24%
# Class 2: 21 out of 525: 4.00%
# Class 3: 143 out of 525: 27.24%
