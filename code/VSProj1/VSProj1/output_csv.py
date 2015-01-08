import pandas as pd
import os
import shutil
import pre_process

def write_meta_csv(clf_name, df_output, Y_test, del_folder = False):
    df_output[clf_name] = Y_test
    df_output.rename(columns={'id': '', clf_name: 'Made Donation in March 2007'}, inplace=True)
    write_csv(clf_name, pre_process.clfFolderMeta, df_output, pre_process.test_csv_name, del_folder)

def write_test_csv(clf_name, df_output, Y_test, del_folder = False):

    df_output[clf_name] = Y_test
    write_csv(clf_name, pre_process.clfFolderBase, df_output, pre_process.test_csv_name, del_folder) 

def write_base_csv(clf_name, df_output, del_folder = False):
    clfFolder = pre_process.clfFolderBase
    write_csv(clf_name, pre_process.clfFolderBase, df_output, pre_process.base_csv_name, del_folder)

def write_csv(clf_name, clfFolder, df_output, file_name, del_folder = False):
    
    clfFolder = clfFolder + clf_name

    if del_folder:
        shutil.rmtree(clfFolder, ignore_errors=True)
    if not os.path.exists(clfFolder):
        os.makedirs(clfFolder)

    df_output.to_csv(clfFolder +"\\"+ file_name, index = False) 

def write_gs_params_base(clf_name, bp, bs):
    write_gs_params(clf_name, pre_process.clfFolderBase, bp, bs)

def write_gs_params_meta(clf_name, bp, bs):
    write_gs_params(clf_name, pre_process.clfFolderMeta, bp, bs)

def write_gs_params(clf_name, clfFolder, bp, bs):
    clfFolder = clfFolder + clf_name
    score_file = open(clfFolder + "\\Score.txt", "w")
    score_file.write("gs.best_params_ = {0}, gs.best_score_ = {1}".format(bp, bs))
    score_file.close()
    