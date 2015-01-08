import pandas as pd
import output_csv
import pre_process


def run_meta_clf(clf, df_output, X_train, Y_train, predict_method = ''):

    clf.fit(X_train, Y_train)
        
    if predict_method == 'proba':
        predict = clf.predict_proba(X_test)[:,1]
    elif predict_method == 'decision':
        predict = clf.decision_function(X_test)
    else:
        predict = clf.predict(X_test)

    df_output.loc[df_testx_split.index.values, clf.__class__.__name__] = predict

    output_csv.write_base_csv(clf.__class__.__name__, df_output)


if __name__ == '__main__':

    pp_meta = pre_process.PreProcessMeta()