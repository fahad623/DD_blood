
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import cross_val
import pre_process
import output_csv
from sklearn.metrics import log_loss

predict_method = 'proba'

def make_best_classifier():
    return RandomForestClassifier(n_jobs = 7, n_estimators = 65, max_depth = 3), predict_method

def train_base_clf(pp):
    clf = make_best_classifier()[0]
    n_estimators_range = np.arange(40, 70, 5)
    max_depth_range = np.arange(2, 4, 1)

    param_grid = dict(n_estimators = n_estimators_range, max_depth = max_depth_range)
    clf, bp, bs = cross_val.fit_clf(clf, pp.X_train, pp.Y_train, param_grid, 'log_loss')
    output_csv.write_test_csv(clf.__class__.__name__, pp.df_output_test, clf.predict_proba(pp.X_test)[:,1])
    output_csv.write_gs_params_base(clf.__class__.__name__, bp, bs, 
                                    clf.score(pp_base.X_train, pp_base.Y_train), log_loss(pp.Y_train, clf.predict_proba(pp.X_train)[:,1]))
    return clf, predict_method


if __name__ == '__main__':
    pp_base = pre_process.PreProcessBase(normalize = False)
    clf = train_base_clf(pp_base)[0]