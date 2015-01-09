
from sklearn.ensemble import AdaBoostClassifier
import numpy as np
import cross_val
import pre_process
import output_csv

predict_method = 'proba'

def make_best_classifier():
    return AdaBoostClassifier(n_estimators = 70, learning_rate = 1.0), predict_method

def train_base_clf(pp):
    clf = make_best_classifier()[0]
    n_estimators_range = np.arange(50, 200, 50)
    learning_rate_range = np.logspace(-1, 1, 2, endpoint = True)
    param_grid = dict(n_estimators = n_estimators_range, learning_rate = learning_rate_range)
    clf, bp, bs = cross_val.fit_clf(clf, pp.X_train, pp.Y_train, param_grid)
    output_csv.write_test_csv(clf.__class__.__name__, pp.df_output_test, clf.predict_proba(pp.X_test)[:,1])
    output_csv.write_gs_params_base(clf.__class__.__name__, bp, bs)
    return clf, predict_method

if __name__ == '__main__':
    pp_base = pre_process.PreProcessBase()
    clf = train_base_clf(pp_base)[0]
    print "Total Score - {0}".format(clf.score(pp_base.X_train, pp_base.Y_train))