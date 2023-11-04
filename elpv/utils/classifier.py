from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV

def grid_search(X_train, y_train, X_test, y_test, estimator, params, cv, scoring):
    clf = GridSearchCV(estimator, params,
                               cv=cv,
                               scoring=scoring,
                               verbose=3,
                               n_jobs=-1)
    clf.fit(X_train, y_train)
    
    best_clf = clf.best_estimator_
    
    pred = best_clf.predict(X_test)
    report = classification_report(y_test, pred, zero_division=1)
    acc = accuracy_score(y_test, pred)
    conf_mat = confusion_matrix(y_test, pred)
    
    print(f'Best params: {clf.best_params_}')
    print('Classification report:')
    print(report)
    print('-'*50)
    print()
    
    return best_clf, pred, acc, conf_mat