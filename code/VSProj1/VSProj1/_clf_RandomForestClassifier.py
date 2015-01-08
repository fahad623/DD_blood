
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import cross_val
import pre_process
import output_csv

def train_base_clf(pp):

    pp = pre_process.PreProcessBase()
    clf = RandomForestClassifier()
    clf, bp, bs = cross_val.fit_clf(clf, pp.X_train, pp.Y_train)
    output_csv.write_test_csv(clf.__class__.__name__, pp.df_output_test, clf.predict_proba(pp.X_test)[:,1])
    output_csv.write_gs_params(clf.__class__.__name__, bp, bs)
    return clf, 'proba'

def make_best_classifier():
    return RandomForestClassifier(), 'proba'

if __name__ == '__main__':
    pp_base = pre_process.PreProcessBase()
    train_base_clf(pp_base)