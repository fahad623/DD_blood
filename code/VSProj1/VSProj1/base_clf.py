import pandas as pd
import output_csv


def run_base_clf(clf, df_output, X_train_full, Y_train_full, n_folds, predict_method = ''):

    kf = KFold(X_train_full.shape[0], n_folds = n_folds)

    for train, test in kf:       
        df_testx_split = df_output.loc[test]
        X_train = X_train_full[train]
        X_test  = X_train_full[test]
        Y_train = Y_train_full[train]

        clf.fit(X_train, Y_train)
        
        if predict_method == 'proba':
            predict = clf.predict_proba(X_test)[:,1]
        elif predict_method == 'decision':
            predict = clf.decision_function(X_test)
        else:
            predict = clf.predict(X_test)

        df_output.loc[df_testx_split.index.values, clf.__class__.__name__] = predict
        del df_testx_split, X_train, X_test, Y_train, predict

    output_csv.write_base_csv(clf, df_output)