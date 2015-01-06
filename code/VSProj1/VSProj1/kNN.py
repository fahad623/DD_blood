import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.grid_search import GridSearchCV
from sklearn import preprocessing
import os
import shutil

trainFile = "..\\..\\..\\data\\Warm_Up_Predict_Blood_Donations_-_Traning_Data.csv"
testFile = "..\\..\\..\\data\\Warm_Up_Predict_Blood_Donations_-_Test_Data.csv"
clfFolder = "..\\..\\..\\classifier\\kNN\\"

def cv_optimize(X_train, Y_train, clf):
    n_neighbors_range = np.arange(5, 51, 2)
    param_grid = dict(n_neighbors = n_neighbors_range)

    gs = GridSearchCV(clf, param_grid = param_grid, cv = 100, n_jobs = 4, verbose = 3)
    gs.fit(X_train, Y_train)
    print "gs.best_params_ = {0}, gs.best_score_ = {1}".format(gs.best_params_, gs.best_score_)
    return gs.best_estimator_, gs.best_params_, gs.best_score_

def fit_clf(X_train, Y_train):
    clf = KNeighborsClassifier(n_neighbors = 31)
    clf, bp, bs = cv_optimize(X_train, Y_train, clf)    
    clf.fit(X_train, Y_train)
    return clf


def normalize_data(X_train, X_test):
    
    scaler = preprocessing.StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test

if __name__ == '__main__':

    shutil.rmtree(clfFolder, ignore_errors=True)

    df_train= pd.read_csv(trainFile)
    X_train = df_train.ix[:, 1:5].astype(np.float64).values
    Y_train = df_train.ix[:, 5].astype(np.float64).values

    df_test= pd.read_csv(testFile)
    X_test = df_test.ix[:, 1:5].astype(np.float64).values

    X_train, X_test = normalize_data(X_train, X_test)

    df_output = pd.DataFrame(df_test.ix[:,0])

    clf = fit_clf(X_train, Y_train)

    df_output['Made Donation in March 2007'] = clf.predict_proba(X_test)[:,1]

    if not os.path.exists(clfFolder):
        os.makedirs(clfFolder)

    df_output.to_csv(clfFolder + "output.csv", index = False) 

