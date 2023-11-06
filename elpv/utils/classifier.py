from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV
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

    value, _ = np.unique(y_train, return_counts=True)
    print(f'Class: {value} Val Score: {clf.best_score_:.2f} use {clf.best_params_}')

    mean_train_score = clf.cv_results_['mean_train_score']
    mean_test_score = clf.cv_results_['mean_test_score']
    params = clf.cv_results_['params']
    
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