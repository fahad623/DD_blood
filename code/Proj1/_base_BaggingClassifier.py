from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
import numpy as np
import cross_val
import pre_process
import output_csv
from sklearn.metrics import log_loss

predict_method = 'proba'

def make_best_classifier():
    return BaggingClassifier(KNeighborsClassifier(n_neighbors = 31, algorithm = 'brute'), n_estimators = 50, max_samples=1.0, max_features=1.0, n_jobs = 7), predict_method

def train_base_clf(pp):
    clf = make_best_classifier()[0]
    n_estimators_range = np.arange(10, 160, 10)
    param_grid = dict(n_estimators= n_estimators_range)
    clf, bp, bs = cross_val.fit_clf(clf, pp.X_train, pp.Y_train, {}, 'log_loss')
    output_csv.write_test_csv(clf.__class__.__name__, pp.df_output_test, clf.predict_proba(pp.X_test)[:,1])
    output_csv.write_gs_params_base(clf.__class__.__name__, bp, bs, 
                                    clf.score(pp_base.X_train, pp_base.Y_train), log_loss(pp.Y_train, clf.predict_proba(pp.X_train)[:,1]))
    return clf, predict_method


if __name__ == '__main__':
    pp_base = pre_process.PreProcessBase()
    clf = train_base_clf(pp_base)[0]
