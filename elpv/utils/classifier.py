from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV

def grid_search(X_train, y_train, X_test, y_test, estimator, params, cv, scoring, verbose=3, n_jobs=-1, return_train_score=True):
    clf = GridSearchCV(estimator, params,
                               cv=cv,
                               scoring=scoring,
                               verbose=verbose,
                               n_jobs=n_jobs,
                               return_train_score=return_train_score)
    clf.fit(X_train, y_train)
    
    best_clf = clf.best_estimator_
    print(f'Best Score: {clf.best_score_} use {clf.best_params_}')
    
    result = clf.cv_results_
    
    # Test prediction
    pred = best_clf.predict(X_test)

    acc, conf_mat = get_report(y_test, pred)
        
    
    return result, best_clf, pred, acc, conf_mat

def get_report(y_test, pred):
    report = classification_report(y_test, pred, zero_division=0)
    acc = accuracy_score(y_test, pred)
    conf_mat = confusion_matrix(y_test, pred)
    
    print('Testset Classification Report:')
    print(report)
    print()
    
    return  acc, conf_mat