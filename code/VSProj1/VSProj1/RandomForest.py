
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import cross_val
import pre_process
import output_csv

if __name__ == '__main__':

    pp = pre_process.PreProcess()
    clf = RandomForestClassifier()
    clf = cross_val.cv_optimize(clf, pp.X_train, pp.Y_train)
    output_csv.write_test_csv(clf, pp.df_output, clf.predict_proba(pp.X_test)[:,1])

