import pandas as pd
import pre_process

clf_names_avg = ['KNeighborsClassifier', 'XGBoostClassifier', 'LogisticRegression']

def avg_to_output():
    
    df_list_test = []
    for i, item in enumerate(clf_names_avg):
        path_test = pre_process.clfFolderBase + item + '/' + pre_process.test_csv_name

        test1 = pd.read_csv(path_test)
        if i > 0:
            test1.drop(['id'], axis=1, inplace = True)

        df_list_test.append(test1)


    df_test = df_list_test[0].join(df_list_test[1:])
    df_test['Made Donation in March 2007'] = df_test.ix[:, 1:len(clf_names_avg)+1].mean(axis = 1)
    df_test.drop(clf_names_avg, axis = 1, inplace=True)
    df_test.rename(columns={'id': ''}, inplace=True)

    df_test.to_csv(pre_process.clfFolderTop + "avg_test.csv", index = False)

if __name__ == '__main__':
    avg_to_output()