from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, ConfusionMatrixDisplay, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt


def get_report(y_test, pred, name = None):
 
    print('-'*100)
    if name is not None:
        print(f'Classification report for {name}')
    
    report = classification_report(y_test, pred, zero_division=0)
    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred, average='macro')

    print(report)
    print()
    ConfusionMatrixDisplay.from_predictions(y_test, pred)
    plt.show()
    print()
    
    return acc, f1

def one_vs_other_up_sampling(X_train, y_train,total_classes, grid_param, knn, n_neighbors=5):
    clfs = []
    results = []
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
        
        
        result, best = grid_search( X_train_selected, y_train_selected, **grid_param)
        clfs.append(best)
        results.append(result)
        
    return results, clfs

def down_sampling(total_classes, X_train, y_train, grid_param):
    
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

            _, best = grid_search( X_train_selected, y_train_selected, **grid_param)
            clfs.append(best)
            
    return clfs
    
def vote(X_test, clfs):
    predictions = []
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
        for _ in range(percentage):
     
            selected = np.random.choice(nnarray, 1)
            w = np.random.rand()
            new = data + w * (minority[selected[0]] - data)
            new = new.astype(int)
            # print(f'Target: {data}, knn: {nnarray},selected index: {selected}, Selected neighbor: {minority[selected[0]]}, New: {new}')
            synthetic.append(new)
            counter += 1 
    print(f'Generated {counter} synthetic samples.')
    synthetic = np.array(synthetic)
    return synthetic

def up_sampling(X_train, y_train, total_classes, grid_param, knn, n_neighbors=5):
    
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
                synthetic = smote(X_train[negative], np.zeros(len(negative)), percentage=len(postive)/len(negative)*100, clf = knn, n_neighbors=n_neighbors)
                synthetic_y = np.ones(len(synthetic)) * j
                
            elif len(negative) > len(postive):
                synthetic = smote(X_train[postive], np.zeros(len(postive)), percentage=len(negative)/len(postive)*100, clf = knn, n_neighbors=n_neighbors)
                synthetic_y = np.ones(len(synthetic)) * i
                
            X_train_selected = X_train[np.concatenate((postive, negative))]
            y_train_selected = y_train[np.concatenate((postive, negative))]
            
            X_train_selected = np.concatenate((X_train_selected, synthetic))
            y_train_selected = np.concatenate((y_train_selected, synthetic_y))
            
 
            result, best = grid_search( X_train_selected, y_train_selected, **grid_param)
            clfs.append(best)
    return clfs

def grid_search(X_train, y_train, 
                estimator, model_params, scoring, refit= False,
                verbose=3, n_jobs=-1, cv=3, return_train_score=True,
                k = None, classes = None, logger = None):
    
    clf = GridSearchCV(estimator(), model_params,
                               cv=cv,
                               scoring=scoring, refit = refit,
                               verbose=verbose,
                               n_jobs=n_jobs,
                               return_train_score=return_train_score)
    clf.fit(X_train, y_train)

    best_clf = clf.best_estimator_
    
    value, _ = np.unique(y_train, return_counts=True)
    print(f'Class: {value} Val Score: {clf.best_score_:.2f} use {clf.best_params_}')
    

    return clf.cv_results_, best_clf




def distance_vote(X_test, clfs):
    
    distance =  np.vstack([clf.decision_function(X_test) for clf in clfs])
    pred = np.argmax(distance, axis = 0)

    
    return distance, pred

def print_misclassifications(y_test, pred, distance):
    distance = np.round(distance, 2)
    misclassified_indices = np.where(pred != y_test)[0]
    success_count = 0
    fail_count = 0

    for index in misclassified_indices:
        true_class_score = distance[y_test[index], index]
        predicted_class_score = distance[pred[index], index]
        if true_class_score > 0 and true_class_score > predicted_class_score:
            outcome = "Success"
            success_count += 1
        else:
            outcome = "Failed"
            fail_count += 1

        print(f"Index {index}:")
        print(f"True: {y_test[index]}")
        print(f"Predicted: {pred[index]}")
        print(f"Scores {distance[:, index]}")
        print(f"Outcome {outcome}")
        print()

    print(f"Successful: {success_count}")
    print(f"Failed : {fail_count}")
    
    
def reconstruct_pred(errors, thresh_mean, thresh_std, size, isBinary = False):
    prediction = np.zeros(size)
    
    
    if isBinary:
        prediction[errors > thresh_mean + 1* thresh_std] = 1
    else:
        prediction[errors > thresh_mean + 1* thresh_std] = 1
        prediction[errors > thresh_mean + 2* thresh_std] = 2
        prediction[errors > thresh_mean + 3* thresh_std] = 3
    
    return prediction

def mono_poly_split(X, y):
    mono_X = X[y < 4]
    poly_X = X[y > 3]
    
    mono_y = y[y < 4]
    poly_y = y[y > 3]
    
    return mono_X, poly_X, mono_y, poly_y-4

def combine_mono_poly(mono_X, poly_X, mono_y, poly_y):
    X = np.concatenate((mono_X, poly_X))
    y = np.concatenate((mono_y, poly_y))
    return X, y