from sklearn.linear_model import LogisticRegression
import numpy as np
import cross_val
import pre_process
import output_csv
from sklearn.metrics import log_loss

predict_method = 'proba'

def make_best_classifier():
    #return LogisticRegression(C = 10, tol = 1e-4), predict_method
    return LogisticRegression(C = 0.9, tol = 1e-11), predict_method

def train_base_clf(pp):
    clf = make_best_classifier()[0]
    #C_range = 10.0 ** np.arange(-3, 3)
    C_range = np.arange(0.1,1,0.1)
    tol_range = 10.0 ** np.arange(-15, -7)
    param_grid = dict(C = C_range, tol = tol_range)
    clf, bp, bs = cross_val.fit_clf(clf, pp.X_train, pp.Y_train, param_grid, 'log_loss')
    output_csv.write_test_csv(clf.__class__.__name__, pp.df_output_test, clf.predict_proba(pp.X_test)[:,1])
    output_csv.write_gs_params_base(clf.__class__.__name__, bp, bs, 
                                    clf.score(pp_base.X_train, pp_base.Y_train), log_loss(pp.Y_train, clf.predict_proba(pp.X_train)[:,1]))
    return clf, predict_method


if __name__ == '__main__':
    pp_base = pre_process.PreProcessBase()
    clf = train_base_clf(pp_base)[0]