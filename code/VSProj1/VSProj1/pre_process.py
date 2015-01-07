import pandas as pd
import numpy as np
from sklearn import preprocessing

trainFile = "..\\..\\..\\data\\Warm_Up_Predict_Blood_Donations_-_Traning_Data.csv"
testFile = "..\\..\\..\\data\\Warm_Up_Predict_Blood_Donations_-_Test_Data.csv"
clfFolder = "..\\..\\..\\classifier\\"
base_csv_name = "base_output.csv"
test_csv_name = "test_output.csv"


class PreProcess(object):   
    
    out_col_name = "Made Donation in March 2007"

    def __init__(self):
        self.load()
    
    def clean(self):
        pass

    def normalize(self):
        scaler = preprocessing.StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)

    def load(self):
        df_train = pd.read_csv(trainFile)
        df_test= pd.read_csv(testFile)
        self.df_output = pd.DataFrame(df_test.ix[:,0])

        self.X_train = df_train.ix[:, 1:5].astype(np.float64).values
        self.Y_train = df_train.ix[:, 5].astype(np.float64).values
        self.X_test = df_test.ix[:, 1:5].astype(np.float64).values

        del df_train, df_test
        self.normalize()

    def get_train_test(self):
        return self.X_train, self.X_test, self.Y_train
