from sklearn.svm import LinearSVC
import numpy as np

from .. import cross_val
import pre_process
import output_csv

if __name__ == '__main__':

    pp = pre_process.PreProcess()
    clf = LinearSVC()
    C_range = 10.0 ** np.arange(-12, 1)
    param_grid = dict(C=C_range)
    clf = cross_val.fit_clf(clf, pp.X_train, pp.Y_train)
    output_csv.write_test_csv(clf, pp.df_output, clf.decision_function(pp.X_test))

