from sklearn.ensemble import RandomForestClassifier
import numpy as np
import cross_val
import pre_process
import output_csv

predict_method = 'proba'


def make_best_classifier():
    return RandomForestClassifier(n_jobs = 7, n_estimators = 170), predict_method

def train_meta_clf(pp_base, pp_meta):    
    clf = make_best_classifier()[0]
    n_estimators_range = np.arange(100, 200, 10)
    param_grid = dict(n_estimators = n_estimators_range)
    clf, bp, bs = cross_val.fit_clf(clf, pp_meta.X_train, pp_base.Y_train, param_grid, 'log_loss')
    output_csv.write_meta_csv(clf.__class__.__name__, pp_base.df_output_test, clf.predict_proba(pp_meta.X_test)[:,1]) 
    output_csv.write_gs_params_meta(clf.__class__.__name__, bp, bs)
    return clf, predict_method


if __name__ == '__main__':
    pp_base = pre_process.PreProcessBase()
    pp_meta = pre_process.PreProcessMeta()
    train_meta_clf(pp_base, pp_meta)
    

