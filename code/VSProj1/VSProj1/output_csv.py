import pandas as pd
import os
import shutil
import pre_process

def write_test_csv(clf, df_output, Y_test, del_folder = False):

    df_output['Made Donation in March 2007'] = Y_test
    write_csv(clf, df_output, pre_process.test_csv_name, del_folder) 

def write_base_csv(clf, df_output, del_folder = False):

    write_csv(clf, df_output, pre_process.base_csv_name, del_folder)

def write_csv(clf, df_output, file_name, del_folder = False):
    
    clfFolder = pre_process.clfFolder + clf.__class__.__name__

    if del_folder:
        shutil.rmtree(clfFolder, ignore_errors=True)
    if not os.path.exists(clfFolder):
        os.makedirs(clfFolder)

    df_output.to_csv(clfFolder +"\\"+ file_name, index = False) 
