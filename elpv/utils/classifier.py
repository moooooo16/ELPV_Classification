from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV
from scipy import stats
import numpy as np


def grid_search(X_train, y_train, 
                estimator, params, scoring, 
                k, classes, logger,
                verbose=0, n_jobs=-1, cv=5, 
                return_train_score=True):
    
    clf = GridSearchCV(estimator, params,
                               cv=cv,
                               scoring=scoring,
                               verbose=verbose,
                               n_jobs=n_jobs,
                               return_train_score=return_train_score)
    clf.fit(X_train, y_train)

    best_clf = clf.best_estimator_
    mean_train_score = clf.cv_results_['mean_train_score']
    mean_test_score = clf.cv_results_['mean_test_score']
    params = clf.cv_results_['params']
    
    
  
    value, _ = np.unique(y_train, return_counts=True)
    print(f'Class: {value} Val Score: {clf.best_score_:.2f} use {clf.best_params_}')
    
    if logger:
        if k is None and classes is None:
             logger.info(f'{clf.best_score_:.2f},{clf.best_params_},{np.mean(mean_train_score):.2f},{np.mean(mean_test_score):.2f}')
        elif k is None and classes is not None:
             logger.info(f'{classes[0]},{classes[1]},{clf.best_score_:.2f},{clf.best_params_},{np.mean(mean_train_score):.2f},{np.mean(mean_test_score):.2f}')
        elif k is not None and classes is not None:
             logger.info(f'{k},{classes[0]},{classes[1]},{clf.best_score_:.2f},{clf.best_params_},{np.mean(mean_train_score):.2f},{np.mean(mean_test_score):.2f}')

    return best_clf

def get_report(y_test, pred):
    report = classification_report(y_test, pred, zero_division=0)
    acc = accuracy_score(y_test, pred)
    conf_mat = confusion_matrix(y_test, pred)
    
    print('Testset Classification Report:')
    print(report)
    print()
    
    return  acc, conf_mat

def down_sampling(total_classes, X_train, y_train, clf, param,  k, logger=None):
    
    clfs = []
    
    for i in total_classes:
        for j in total_classes:

            if j <= i:
                continue
            
            
            postive = np.where(y_train == i)[0]
            negative = np.where(y_train == j)[0]
            
            if len(postive) > len(negative):
                postive = np.random.choice(postive, len(negative), replace=False)
            elif len(postive) < len(negative):
                negative = np.random.choice(negative, len(postive), replace=False)

            
            X_train_selected = X_train[np.concatenate([postive, negative])]
            y_train_selected = y_train[np.concatenate([postive, negative])]
            svm = clf()
            best = grid_search( X_train_selected, y_train_selected, svm, param, k = k, classes = (i, j), scoring='accuracy', logger = logger)
            clfs.append(best)
            
    return clfs
    
def vote(X_test, clfs, predictions):
    for clf in clfs:
        pred = clf.predict(X_test)
        predictions.append(pred)
    predictions = np.array(predictions)
    votes, counts = stats.mode(predictions, axis=0)
        
    
    return predictions, votes, counts

def smote(minority, y_minority, percentage, clf, n_neighbors=5):

    synthetic = []
    counter = 0
    if percentage < 100:
        minority = np.random.choice(minority, int(len(minority) * percentage / 100), replace=False)
        percentage = 100

        
    percentage = int(percentage / 100)
    
    for data in minority:
        
        knn = clf(n_neighbors=n_neighbors).fit(minority, y_minority)
  
        _, nnarray = knn.kneighbors(data.reshape(1,-1), n_neighbors+1)
        nnarray = nnarray[0][1:] 
        for i in range(percentage):
     
            selected = np.random.choice(nnarray, 1)
            w = np.random.rand()
            new = data + w * (minority[selected[0]] - data)

            # print(f'Target: {data}, knn: {nnarray},selected index: {selected}, Selected neighbor: {minority[selected[0]]}, New: {new}')
            synthetic.append(new)
            counter += 1 
    print(f'Generated {counter} synthetic samples.')
    synthetic = np.array(synthetic)
    return synthetic

def up_sampling(total_classes, X_train, y_train, clf, param, k, logger, knn):
    
    clfs = []
    for i in total_classes:
        for j in total_classes:
            if j <= i:
                continue
            postive = np.where(y_train == i)[0]
            negative = np.where(y_train == j)[0]
            synthetic = None
            synthetic_y = None
            
            if len(postive) > len(negative):
                synthetic = smote(X_train[negative], np.zeros(len(negative)), percentage=len(postive)/len(negative)*100, clf = knn, n_neighbors=5)
                synthetic_y = np.ones(len(synthetic)) * j
                
            elif len(negative) > len(postive):
                synthetic = smote(X_train[postive], np.zeros(len(postive)), percentage=len(negative)/len(postive)*100, clf = knn, n_neighbors=5)
                synthetic_y = np.ones(len(synthetic)) * i
                
            X_train_selected = X_train[np.concatenate((postive, negative))]
            y_train_selected = y_train[np.concatenate((postive, negative))]
            
            X_train_selected = np.concatenate((X_train_selected, synthetic))
            y_train_selected = np.concatenate((y_train_selected, synthetic_y))
            
            svm = clf()
            best = grid_search( X_train_selected, y_train_selected, svm, param, k = k, classes = (i, j), scoring='accuracy', logger = logger)
            clfs.append(best)
    return clfs

def one_vs_other_up_sampling(total_classes, X_train, y_train, clf, param, k, logger, knn, n_neighbors=5):
    clfs = []
    for i in total_classes:
        
        synthetic = []
        synthetic_y = []
        binary_y_train = np.where(y_train == i, i, -1)
        
        postive = np.where(binary_y_train == i)[0]
        negative = np.where(binary_y_train == -1)[0]
        
        # positive  < negative
        if len(postive) < len(negative):
            
            synthetic = smote(X_train[postive], np.zeros(len(postive)), percentage=len(negative)/len(postive)*100, clf = knn, n_neighbors=n_neighbors)
            synthetic_y = np.ones(len(synthetic)) * i
        
        X_train_selected = X_train[np.concatenate((postive, negative))]
        y_train_selected = binary_y_train[np.concatenate((postive, negative))]
        
        if len(synthetic) > 0:
            X_train_selected = np.concatenate((X_train_selected, synthetic))
            y_train_selected = np.concatenate((y_train_selected, synthetic_y))
        
        print(f'Trainig on: {np.unique(y_train_selected, return_counts=True)}')
        
        svm = clf()
        best = grid_search( X_train_selected, y_train_selected, svm, param, k = k, classes = (i,-1), scoring='accuracy', logger = logger)
        clfs.append(best)
        
    return clfs

def distance_vote(X_test, clfs, predictions):
    
    distance =  np.vstack([clf.decision_function(X_test) for clf in clfs])
    pred = np.argmax(distance, axis = 0)

    
    return distance, pred
