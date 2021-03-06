from sklearn.svm import LinearSVC
import numpy as np
import cross_val
import pre_process
import output_csv

predict_method = 'decision'

def make_best_classifier():
    return LinearSVC(C = 0.003) , predict_method
    #return LinearSVC(C = 1e-5) , predict_method

def train_base_clf(pp):
    clf = make_best_classifier()[0]
    C_range = np.arange(0.001, 0.010, 0.001)
    param_grid = dict(C=C_range)
    clf, bp, bs = cross_val.fit_clf(clf, pp.X_train, pp.Y_train, param_grid )
    #clf, bp, bs = cross_val.fit_clf(clf, pp.X_train, pp.Y_train, param_grid)
    output_csv.write_test_csv(clf.__class__.__name__, pp.df_output_test, clf.decision_function(pp.X_test))
    output_csv.write_gs_params_base(clf.__class__.__name__, bp, bs, clf.score(pp.X_train, pp.Y_train))
    return clf, predict_method


if __name__ == '__main__':
    pp_base = pre_process.PreProcessBase()
    train_base_clf(pp_base)