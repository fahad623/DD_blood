from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
import cross_val
import pre_process
import output_csv

predict_method = 'proba'


def make_best_classifier():
    return GradientBoostingClassifier(max_depth = 2, n_estimators = 500, learning_rate = 0.01), predict_method

def train_base_clf(pp):    
    clf = make_best_classifier()[0]
    max_depth_range = np.arange(2, 5)
    n_estimators_range = np.arange(200, 700, 50)
    learning_rate_range = np.logspace(-3, -1, 7, endpoint = True)
    param_grid = dict(max_depth=max_depth_range, n_estimators= n_estimators_range, learning_rate = learning_rate_range)
    clf, bp, bs = cross_val.fit_clf(clf, pp.X_train, pp.Y_train, param_grid, 'log_loss')
    output_csv.write_test_csv(clf.__class__.__name__, pp.df_output_test, clf.predict_proba(pp.X_test)[:,1])
    output_csv.write_gs_params_base(clf.__class__.__name__, bp, bs, clf.score(pp.X_train, pp.Y_train))
    return clf, predict_method


if __name__ == '__main__':
    pp_base = pre_process.PreProcessBase()
    train_base_clf(pp_base)
    

