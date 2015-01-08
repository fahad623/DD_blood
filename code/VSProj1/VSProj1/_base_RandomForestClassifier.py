
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import cross_val
import pre_process
import output_csv

predict_method = 'proba'

def make_best_classifier():
    return RandomForestClassifier(n_jobs = 7, n_estimators = 110), predict_method

def train_base_clf(pp):
    clf = make_best_classifier()[0]
    n_estimators_range = np.arange(100, 200, 10)
    param_grid = dict(n_estimators = n_estimators_range)
    clf, bp, bs = cross_val.fit_clf(clf, pp.X_train, pp.Y_train, {})
    output_csv.write_test_csv(clf.__class__.__name__, pp.df_output_test, clf.predict_proba(pp.X_test)[:,1])
    output_csv.write_gs_params_base(clf.__class__.__name__, bp, bs)
    return clf, predict_method


if __name__ == '__main__':
    pp_base = pre_process.PreProcessBase()
    train_base_clf(pp_base)