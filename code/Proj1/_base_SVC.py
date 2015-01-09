from sklearn.svm import SVC
import numpy as np
import cross_val
import pre_process
import output_csv

predict_method = 'decision'


def make_best_classifier():
    return SVC(C = 1.0, gamma = 1.0), predict_method

def train_base_clf(pp):
    clf = make_best_classifier()[0]
    C_range = 10.0 ** np.arange(-12, 1)
    gamma_range = 10.0 ** np.arange(-12, 1)
    param_grid = dict(C=C_range, gamma = gamma_range)
    clf, bp, bs = cross_val.fit_clf(clf, pp.X_train, pp.Y_train, {})
    output_csv.write_test_csv(clf.__class__.__name__, pp.df_output_test, clf.decision_function(pp.X_test)) 
    output_csv.write_gs_params_base(clf.__class__.__name__, bp, bs)
    return clf, predict_method


if __name__ == '__main__':
    pp_base = pre_process.PreProcessBase()
    clf = train_base_clf(pp_base)[0]
    print "Total Score - {0}".format(clf.score(pp_base.X_train, pp_base.Y_train))
