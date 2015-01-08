from sklearn.svm import SVC
import numpy as np
import cross_val
import pre_process
import output_csv

def train_base_clf(pp):

    pp = pre_process.PreProcessBase()
    clf = SVC()
    C_range = 10.0 ** np.arange(-12, 1)
    gamma_range = 10.0 ** np.arange(-12, 1)
    param_grid = dict(C=C_range, gamma = gamma_range)
    clf, bp, bs = cross_val.fit_clf(clf, pp.X_train, pp.Y_train, param_grid)
    output_csv.write_test_csv(clf.__class__.__name__, pp.df_output_test, clf.decision_function(pp.X_test)) 
    output_csv.write_gs_params(clf.__class__.__name__, bp, bs)
    return clf, 'decision'

def make_best_classifier():
    return SVC(C = 1.0, gamma = 1.0), 'decision'

if __name__ == '__main__':
    pp_base = pre_process.PreProcessBase()
    train_base_clf(pp_base)

