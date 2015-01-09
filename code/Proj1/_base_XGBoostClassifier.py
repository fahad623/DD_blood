import sys
sys.path.append('../../../../xgboost/wrapper/')
import xgboost as xgb
import numpy as np
import cross_val
import pre_process
import output_csv

predict_method = ''

class XGBoostClassifier(object):
    param = {}
    # use logistic regression for binary classification, output probability
    param['objective'] = 'binary:logistic'
    param['eta'] = 0.2
    param['max_depth'] = 1000
    param['subsample'] = 0.8
    param['silent'] = 1
    param['nthread'] = 7
    num_round = 1500

    def __init__(self):
        pass        

    def fit(self, X_train, Y_train):
        xg_train = xgb.DMatrix(X_train, label=Y_train)   
        watchlist = [ (xg_train,'train'), (xg_test, 'test') ]     
        self.bst = xgb.train( XGBoostClassifier.param, xg_train, XGBoostClassifier.num_round)   

    def predict(self, X_test):
        xg_test = xgb.DMatrix(X_test)
        return self.bst.predict( xg_test )


def make_best_classifier():
    return XGBoostClassifier(), predict_method

def train_base_clf(pp):    
    clf = make_best_classifier()[0]
    clf.fit(pp.X_train, pp.Y_train)
    output_csv.write_test_csv(clf.__class__.__name__, pp.df_output_test, clf.predict(pp.X_test))
    output_csv.write_gs_params_base(clf.__class__.__name__)
    return clf, predict_method


if __name__ == '__main__':
    pp_base = pre_process.PreProcessBase()
    train_base_clf(pp_base)
    

