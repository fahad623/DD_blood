from sklearn.linear_model import LogisticRegression
import numpy as np
import cross_val
import pre_process
import output_csv
from sklearn.metrics import log_loss

predict_method = 'proba'

def make_best_classifier():
    return LogisticRegression(C = 17, tol = 1e-7, random_state = 29), predict_method

def train_meta_clf(pp_base, pp_meta):
    clf = make_best_classifier()[0]
    #C_range = 10.0 ** np.arange(-3, 3)
    #tol_range = 10.0 ** np.arange(-7, -3)

    C_range = np.arange(1, 25, 1)
    tol_range = 10.0 ** np.arange(-15, -5)
    param_grid = dict(C = C_range)
    clf, bp, bs = cross_val.fit_clf(clf, pp_meta.X_train, pp_base.Y_train, param_grid, 'log_loss')
    output_csv.write_meta_csv(clf.__class__.__name__, pp_base.df_output_test, clf.predict_proba(pp_meta.X_test)[:,1]) 
    output_csv.write_gs_params_meta(clf.__class__.__name__, bp, bs, log_loss(pp_base.Y_train, clf.predict_proba(pp_meta.X_train)[:,1]))
    return clf, predict_method


if __name__ == '__main__':
    pp_base = pre_process.PreProcessBase()
    pp_meta = pre_process.PreProcessMeta()
    train_meta_clf(pp_base, pp_meta)

