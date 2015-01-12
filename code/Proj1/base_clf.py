import pandas as pd
from sklearn.cross_validation import KFold
from multiprocessing import Process
import output_csv
import pre_process

base_clf_module_names = ['_base_'+ name for name in pre_process.base_clf_names]
base_clf_modules = map(__import__, base_clf_module_names)

from sklearn.neighbors import KNeighborsClassifier

def fit_predict(clf, X_train, Y_train, X_test, predict_method):
    clf.fit(X_train, Y_train)
        
    if predict_method == 'proba':
        predict = clf.predict_proba(X_test)[:,1]
    elif predict_method == 'decision':
        predict = clf.decision_function(X_test)
    else:
        predict = clf.predict(X_test)

    return predict


def run_base_clf(clf, df_output, X_train_full, Y_train_full, n_folds, predict_method = ''):

    kf = KFold(X_train_full.shape[0], n_folds = n_folds, shuffle = True)

    k = 1
    for train, test in kf:
        print "Running base classifier - {0}, fold - {1}".format(clf.__class__.__name__, k)
        k += 1   
        df_testx_split = df_output.loc[test]
        X_train = X_train_full[train]
        X_test  = X_train_full[test]
        Y_train = Y_train_full[train]

        predict = fit_predict(clf, X_train, Y_train, X_test, predict_method)
        df_output.loc[df_testx_split.index.values, clf.__class__.__name__] = predict
        del df_testx_split, X_train, X_test, Y_train, predict

    output_csv.write_base_csv(clf.__class__.__name__, df_output)

def make_base_classifiers():
    clf_list = [module.make_best_classifier() for module in base_clf_modules]
    return clf_list

def run_base_classifiers(clf_list, pp, n_folds, output_test = False):

    for clf in clf_list:
        p = Process(target = run_base_clf , args=(clf[0], pp.df_output_train.copy(), pp.X_train, pp.Y_train, n_folds, clf[1]))
        p.start()
        #run_base_clf(clf[0], pp.df_output_train.copy(), pp.X_train, pp.Y_train, n_folds, clf[1])
        if output_test:
            predict = fit_predict(clf[0], pp.X_train, pp.Y_train, pp.X_test, clf[1])
            output_csv.write_test_csv(clf[0].__class__.__name__, pp.df_output_test.copy(), predict)
        

if __name__ == '__main__':

    pp = pre_process.PreProcessBase()
    clf_list = make_base_classifiers()
    run_base_classifiers(clf_list, pp, n_folds = 576, output_test = True)