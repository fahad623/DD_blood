from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
import cross_val
import pre_process
import output_csv
from sklearn.metrics import log_loss

predict_method = 'proba'


def make_best_classifier():
    return GradientBoostingClassifier(max_depth = 2, n_estimators = 390, learning_rate = 0.01), predict_method
    #return GradientBoostingClassifier(max_depth = 2, n_estimators = 350, learning_rate = 0.01), predict_method

def train_base_clf(pp):    
    clf = make_best_classifier()[0]
    max_depth_range = np.arange(2, 4)
    n_estimators_range = np.arange(350, 395, 5)
    learning_rate_range = np.logspace(-3, -2, 7, endpoint = True)
    param_grid = dict(n_estimators= n_estimators_range, learning_rate = learning_rate_range)
    clf, bp, bs = cross_val.fit_clf(clf, pp.X_train, pp.Y_train, param_grid, 'log_loss')
    output_csv.write_test_csv(clf.__class__.__name__, pp.df_output_test, clf.predict_proba(pp.X_test)[:,1])
    output_csv.write_gs_params_base(clf.__class__.__name__, bp, bs, 
                                    clf.score(pp_base.X_train, pp_base.Y_train), log_loss(pp.Y_train, clf.predict_proba(pp.X_train)[:,1]))
    return clf, predict_method


if __name__ == '__main__':
    pp_base = pre_process.PreProcessBase()
    train_base_clf(pp_base)
    

