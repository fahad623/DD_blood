from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
import cross_val
import pre_process
import output_csv

def train_base_clf(pp):    
    clf = GradientBoostingClassifier()
    max_depth_range = np.arange(3, 6)
    param_grid = dict(max_depth=max_depth_range)
    clf, bp, bs = cross_val.fit_clf(clf, pp.X_train, pp.Y_train, param_grid)
    output_csv.write_test_csv(clf.__class__.__name__, pp.df_output_test, clf.predict_proba(pp.X_test)[:,1])
    output_csv.write_gs_params(clf.__class__.__name__, bp, bs)
    return clf, 'proba'

def make_best_classifier():
    return GradientBoostingClassifier(max_depth = 3), 'proba'

if __name__ == '__main__':
    pp_base = pre_process.PreProcessBase()
    train_base_clf(pp_base)
    

