import sys
sys.path.append('../../../../xgboost/wrapper/')
import xgboost as xgb
import numpy as np
from sklearn.metrics import log_loss
from sklearn.base import BaseEstimator, ClassifierMixin
import cross_val
import pre_process
import output_csv

predict_method = 'proba'

#class XGBoostClassifier(object):
#    param = {}
#    # use logistic regression for binary classification, output probability
#    param['objective'] = 'binary:logistic'
#    param['eta'] = 0.5
#    param['max_depth'] = 4
#    param['subsample'] = 1
#    param['silent'] = 0
#    param['nthread'] = 7
#    param['seed'] = 31
#    num_round = 30

#    def __init__(self):
#        pass        

#    def fit(self, X_train, Y_train):
#        xg_train = xgb.DMatrix(X_train, label=Y_train)    
#        self.bst = xgb.train( XGBoostClassifier.param, xg_train, XGBoostClassifier.num_round) 

#    def predict(self, X_test):
#        xg_test = xgb.DMatrix(X_test)
#        return self.bst.predict( xg_test )

#    def cross_val_score(self, X_train, Y_train, n_folds):
#        kf = KFold(X_train.shape[0], n_folds = n_folds, shuffle = True)
#        score = 0.0
#        k = 1
#        for train, test in kf:
#            print "Running fold - {0}".format(k)
#            k += 1 
#            X_train_split = X_train[train]
#            X_test_split = X_train[test]

#            Y_train_split = Y_train[train]
#            Y_test_split = Y_train[test]

#            self.fit(X_train_split, Y_train_split)
#            predict = self.predict(X_test_split)

#            loss = log_loss(Y_test_split, predict)
#            score += loss
#        return score/n_folds

class XGBoostClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, objective = 'binary:logistic', eta = 0.3, max_depth = 6, 
                 subsample = 1.0, num_round = 100):
        self.objective = objective
        self.eta = eta
        self.max_depth = max_depth
        self.subsample = subsample
        self.num_round = num_round    
    
    def _init_params(self):
        self.param = {}
        self.param['objective'] = self.objective
        self.param['eta'] = self.eta
        self.param['max_depth'] = self.max_depth
        self.param['subsample'] = self.subsample
        self.param['silent'] = 1
        self.param['nthread'] = 7
        self.param['seed'] = 31

    def fit(self, X_train, Y_train):
        self._init_params()
        xg_train = xgb.DMatrix(X_train, label=Y_train)    
        self.bst = xgb.train(self.param, xg_train, self.num_round) 

    def predict_proba(self, X_test):
        xg_test = xgb.DMatrix(X_test)
        return self.bst.predict( xg_test )

    def predict(self, X_test):
        xg_test = xgb.DMatrix(X_test)
        predict = self.bst.predict( xg_test )
        predict[predict >= 0.5] = 1
        predict[predict < 0.5] = 0
        return predict


def make_best_classifier():
    return XGBoostClassifier(), predict_method

def train_base_clf(pp):    
    clf = make_best_classifier()[0]
    eta_range = np.arange(0.05, 0.25, 0.01)
    max_depth_range = np.arange(2, 5, 1)
    num_round_range = np.arange(15, 40, 2)
    subsample_range = np.arange(0.5, 0.8, 0.02)

    param_grid = dict(eta = eta_range, max_depth = max_depth_range, num_round = num_round_range, subsample = subsample_range)
    clf, bp, bs = cross_val.fit_clf(clf, pp.X_train, pp.Y_train, param_grid, 'log_loss')
    output_csv.write_test_csv(clf.__class__.__name__, pp.df_output_test, clf.predict_proba(pp.X_test))
    output_csv.write_gs_params_base(clf.__class__.__name__, bp, bs, 
                                    clf.score(pp_base.X_train, pp_base.Y_train), log_loss(pp.Y_train, clf.predict_proba(pp.X_train)))
    return clf, predict_method


if __name__ == '__main__':
    pp_base = pre_process.PreProcessBase()
    train_base_clf(pp_base)
    

