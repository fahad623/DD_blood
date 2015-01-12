from sklearn.ensemble import AdaBoostClassifier
import numpy as np
import cross_val
import pre_process
import output_csv
from sklearn.metrics import log_loss

predict_method = 'proba'


def make_best_classifier():
    return AdaBoostClassifier(n_estimators = 70, learning_rate = 1.0), predict_method

def train_meta_clf(pp_base, pp_meta):    
    clf = make_best_classifier()[0]
    n_estimators_range = np.arange(50, 200, 20)
    learning_rate_range = np.logspace(-1, 1, 7, endpoint = True)
    param_grid = dict(n_estimators = n_estimators_range, learning_rate = learning_rate_range)
    clf, bp, bs = cross_val.fit_clf(clf, pp_meta.X_train, pp_base.Y_train, param_grid, 'log_loss')
    output_csv.write_meta_csv(clf.__class__.__name__, pp_base.df_output_test, clf.predict_proba(pp_meta.X_test)[:,1]) 
    output_csv.write_gs_params_meta(clf.__class__.__name__, bp, bs, log_loss(pp_base.Y_train, clf.predict_proba(pp_meta.X_train)[:,1]))
    return clf, predict_method


if __name__ == '__main__':
    pp_base = pre_process.PreProcessBase()
    pp_meta = pre_process.PreProcessMeta()
    train_meta_clf(pp_base, pp_meta)
    

