import pandas as pd
import output_csv
import pre_process
import importlib
meta_clf_module = importlib.import_module('_meta_'+ pre_process.meta_clf_name)

def run_meta_clf(clf, pp_base, pp_meta, predict_method = ''):

    clf.fit(pp_meta.X_train, pp_base.Y_train)
        
    if predict_method == 'proba':
        predict = clf.predict_proba(pp_meta.X_test)[:,1]
    elif predict_method == 'decision':
        predict = clf.decision_function(pp_meta.X_test)
    else:
        predict = clf.predict(pp_meta.X_test)

    output_csv.write_meta_csv(clf.__class__.__name__, pp_base.df_output_test.copy(), predict) 

def make_meta_classifier():
    clf = meta_clf_module.make_best_classifier()
    return clf

if __name__ == '__main__':

    pp_base = pre_process.PreProcessBase()
    pp_meta = pre_process.PreProcessMeta()

    clf = make_meta_classifier()
    run_meta_clf(clf[0], pp_base, pp_meta, clf[1])