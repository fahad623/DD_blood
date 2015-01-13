import pandas as pd
import numpy as np
from sklearn import preprocessing

trainFile = "../../data/Warm_Up_Predict_Blood_Donations_-_Traning_Data.csv"
testFile = "../../data/Warm_Up_Predict_Blood_Donations_-_Test_Data.csv"
clfFolderTop = "../../classifier/"
clfFolderBase = "../../classifier/Base/"
clfFolderMeta = "../../classifier/Meta/"
base_csv_name = "base_output.csv"
test_csv_name = "test_output.csv"

#base_clf_names = ['AdaBoostClassifier','GradientBoostingClassifier', 'KNeighborsClassifier', 'LinearSVC', 'XGBoostClassifier','RandomForestClassifier', 'SVC']
base_clf_names = ['KNeighborsClassifier', 'GradientBoostingClassifier', 'LinearSVC', 'XGBoostClassifier', 'LogisticRegression','SVC']
meta_clf_name = 'LogisticRegression'

class PreProcessBase(object):   

    def __init__(self, normalize = True):
        self.normalize = normalize
        self.df_output_train = pd.DataFrame()
        self.df_output_test = pd.DataFrame()
        self.load()
     
    def clean(self):
        pass

    def normalize_data(self):
        scaler = preprocessing.StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)

    #def normalize_data(self):
    #    scaler = preprocessing.MinMaxScaler()
    #    self.X_train = scaler.fit_transform(self.X_train)
    #    self.X_test = scaler.transform(self.X_test)

    def load(self):
        df_train = pd.read_csv(trainFile)
        df_test= pd.read_csv(testFile)
        self.df_output_train['id'] = pd.DataFrame(df_train.ix[:,0])
        self.df_output_test['id'] = pd.DataFrame(df_test.ix[:,0])

        self.X_train = df_train.ix[:, [1,2,4]].astype(np.float64).values
        self.Y_train = df_train.ix[:, 5].astype(np.float64).values
        self.X_test = df_test.ix[:, [1,2,4]].astype(np.float64).values

        del df_train, df_test
        if self.normalize:
            self.normalize_data()

    def get_train_test(self):
        return self.X_train, self.X_test, self.Y_train


class PreProcessMeta(object):   

    def __init__(self, ):
        self.load()

    def load(self):

        df_list_train = []
        df_list_test = []
        for i, item in enumerate(base_clf_names):
            path_train = clfFolderBase + item + '/' + base_csv_name
            path_test = clfFolderBase + item + '/' + test_csv_name

            train1 = pd.read_csv(path_train)
            test1 = pd.read_csv(path_test)

            if i > 0:
                train1.drop(['id'], axis=1, inplace = True)
                test1.drop(['id'], axis=1, inplace = True)

            df_list_train.append(train1)
            df_list_test.append(test1)


        df_train = df_list_train[0].join(df_list_train[1:])
        df_test = df_list_test[0].join(df_list_test[1:])

        self.X_train = df_train.ix[:, 1:len(base_clf_names)+1].values
        self.X_test = df_test.ix[:, 1:len(base_clf_names)+1].values

        print self.X_test.shape

        df_train.to_csv(clfFolderTop + "meta_train.csv", index = False)
        df_test.to_csv(clfFolderTop + "meta_test.csv", index = False)

        del df_list_train, df_list_test, df_train, df_test

    def get_train_test(self):
        return self.X_train, self.X_test
