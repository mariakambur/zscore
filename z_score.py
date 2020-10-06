import pandas as pd
import numpy as np
from sys import argv


def calculateTrainMeanStd(dfofcolumns, columnsCount):
    mean = []
    std = []
    for i in range(1, columnsCount+1, 1):
        arr = dfofcolumns.iloc[:, i].to_numpy(dtype=int, copy=False)
        mean_i = np.mean(arr)
        mean.append(mean_i)
        std_i = np.std(arr)
        std.append(std_i)
    return mean, std


def parseTsv(t_file, feauture_code):
    df = pd.read_csv(t_file, usecols=['id_job', 'features'], sep='\t')

    # filter rows based on feature code and don't train/include to the test results other rows
    filtered = df.loc[df['features'].str.startswith(
        feauture_code+",")]
    featuresDF = filtered.iloc[:, -1:]  # second column - all features
    featuresSplitted = featuresDF['features'].str.split(
        ',', expand=True)  # features and code separated

    # get jobIDs as sepate df
    jobs = pd.DataFrame({'id_job': filtered.iloc[:, 0]})  # job_id

    if len(jobs) == 0:
        raise ValueError("there no jobs with feature " + feauture_code)
    
    return featuresSplitted, jobs


def calculateZscore(test_dataset, test_columns, columnsCount, t_mean, t_std, feature_code):
    #fist columns if a feature code
    for i in range(1, columnsCount+1, 1):
        arr = test_columns.iloc[:, i].to_numpy(dtype=int, copy=False)
        zs_arr = []
        for el in arr:
            zs = (el - t_mean[i-1])/t_std[i-1]
            zs_arr.append(zs)
        test_dataset['feature_{}_stand_{}_score'.format(
            feature_code, str(i-1))] = zs_arr


def calculateMaxIndexes(test_dataset, test_columns, feature_code):
    max_indexes = test_columns.iloc[:, 1:].to_numpy(
        dtype=int, copy=False).argmax(axis=1)
    test_dataset['max_feature_{}_index'.format(feature_code)] = max_indexes
    return max_indexes


def calculateAbsDev(test_dataset, test_columns, max_indexes, feature_code):
    arrAbsolute = np.array([])
    for i, index in enumerate(max_indexes):
        arr = test_columns.iloc[:, index+1].to_numpy(dtype=int, copy=False)
        av = np.average(arr)
        arrAbsolute = np.append(arrAbsolute, abs(arr[i] - av))
    test_dataset['max_feature_{}_abs_mean_diff'.format(
        feature_code)] = arrAbsolute


def main(file_train, file_test, normalization, feature_code):

    try:
        # get each feature columns from train.tsv based on feature code
        train_columns, _ = parseTsv(file_train, feature_code)
        # get each feature columns from train.tsv and dataset we will further work with
        test_columns, test_dataset = parseTsv(file_test, feature_code)

        columnsCount = test_columns.iloc[:, 1:].shape[1]

        t_mean, t_std = calculateTrainMeanStd(train_columns, columnsCount)
    
        # calculate z score for each test.tsv feature on train.tsv means and stds
        if normalization == 'z-score':
            calculateZscore(test_dataset, test_columns, columnsCount,
                        t_mean, t_std, feature_code)
        else:
            print("Oops! Other normalization methodes are not implemented yet!")

        # find max index for each feature in test dataset
        max_indexes = calculateMaxIndexes(test_dataset, test_columns, feature_code)
        calculateAbsDev(test_dataset, test_columns, max_indexes, feature_code)

        test_dataset.to_csv("test_proc.tsv", sep='\t',  index=False)
    except ValueError as err:
        print(err.args)

if __name__ == '__main__':

    try:
        file_train = argv[1]
        file_test = argv[2]
        normalization = argv[3]
        feature_code = argv[4]
    except IndexError:
        file_train = 'train.tsv'
        file_test = 'test.tsv'
        normalization = 'z-score'
        feature_code = '2'
        # if no arguments use default
        
    main(file_train, file_test, normalization, feature_code)
