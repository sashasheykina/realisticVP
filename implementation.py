import datetime
import warnings
from numpy import unique
from pandas import DataFrame
from pyreadr import pyreadr
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score
from sklearn.metrics import f1_score, matthews_corrcoef
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
import os
import pandas
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import utils
from classes import Log, Performance, ExperimentResults, LogSpan
from time import time


def save_allY(releases, allY, y_pred, app):
    allY.setPred(y_pred)
    output_dir = utils.get_path("my_allY_" + app)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    filename = releases.test_set_release
    resultfile = releases.test_set_release
    if os.path.exists(resultfile):
        os.remove(resultfile)
    if os.path.exists(filename):
        os.remove(filename)
    rf = DataFrame({'nomefile': allY.file, 'Y_IDEAL': allY.ideal, 'Y_REAL': allY.real, 'Y_PRED': allY.pred})
    rf.to_csv(os.path.join(output_dir, resultfile + ".csv"))


def execute_experiment(n, setting):
    warnings.filterwarnings('always')
    experiment_setting = setting[0]
    releases = setting[1]
    utils.print_space()
    print("Experiment " + str(n))
    print(experiment_setting)
    print(releases)
    if experiment_setting.run == "run_real":
        if experiment_setting.validation == "real_labelling":
            X_train, y_train, X_test, y_ideal, y_real, allY = get_real_labelling_data(experiment_setting.dataset,
                                                                                          releases)
            X_train, y_train = balance(X_train, y_train, experiment_setting.balancing)
            performance, y_pred = experiment_real_labelling(X_train, y_train, X_test, y_ideal, y_real,
                                                            experiment_setting.model)
        save_allY(releases, allY, y_pred, experiment_setting.dataset)
        print_performance(performance)
        log = Log.build(experiment_setting, releases, performance)
    else:
        if experiment_setting.validation == "real_labelling":
            max_date, date_release = get_max_date(experiment_setting.dataset, releases)
            #print("DATA MASSIMA: " + str(max_date))
            if experiment_setting.dataset == "moodle":
                max_date = pandas.to_datetime("2011-11-27 23:52:11")
            ts = 0
            cnt_date = date_release
            span = 0
            print(max_date)
            while cnt_date < max_date:
                X_train, y_train, X_test, y_ideal, y_real = get_timespan_data(experiment_setting.dataset, releases, ts)
                day_span = datetime.timedelta(days=90)
                cnt_date = cnt_date + day_span
                span = span + 1
                ts = ts + 90
                count = 0
                for vul in y_train:
                    if vul == 1:
                        count = count + 1
                percentage_vulnerability = round(count/len(y_train)*100, 1)
                X_train, y_train = balance(X_train, y_train, experiment_setting.balancing)
                performance, y_pred = experiment_real_labelling(X_train, y_train, X_test, y_ideal, y_real,
                                                            experiment_setting.model)

                print_performance(performance)
                log = LogSpan.build(experiment_setting, releases, performance, span, percentage_vulnerability)
                print()
                pandas.DataFrame(log).to_csv('experiments_timespan.csv', mode='a', index=False, header=False)
    return log


def print_performance(performance):
    print("Performance Summary:")
    print("Fit time: " + str(performance.fit_time) + " sec")
    print("Precision: " + str(performance.precision))
    print("Recall: " + str(performance.recall))
    print("Accuracy: " + str(performance.accuracy))
    print("Inspection rate: " + str(performance.inspection_rate))
    print("F1-score: " + str(performance.f1_score))
    print("MCC: " + str(performance.mcc))


def my_scorer(model, X, y, y_real):
    y_pred = model.predict(X)
    cm = confusion_matrix(y, y_pred)
    inspection_rate = 0
    try:
        tn = cm[0, 0]
        fp = cm[0, 1]
        fn = cm[1, 0]
        tp = cm[1, 1]
        inspection_rate = (tp + fp) / (tp + tn + fp + fn)
    except:
        print("Undefined value")
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    accuracy = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    mcc = matthews_corrcoef(y, y_pred)
    return {"my_precision": precision, "my_recall": recall, "my_accuracy": accuracy,
            "my_inspection_rate": inspection_rate, "my_f1_score": f1, "my_mcc": mcc}, y_pred


def balance(X, y, b):
    print_class_distribution(y)
    if b == "undersampling":
        print("Performing undersampling...")
        # define undersample strategy: the majority class will be undersampled to match the minority
        undersample = RandomUnderSampler(sampling_strategy='majority')
        try:
            X, y = undersample.fit_resample(X, y)
        except:
            print("Y is all 0")
        print_class_distribution(y)
        return X, y
    elif b == "oversampling":
        print("Performing oversampling...")
        # define oversample strategy: Synthetic Minority Oversampling Technique
        # the minority class will be oversampled to match the majority
        oversample = SMOTE()
        try:
            X, y = oversample.fit_resample(X, y)
        except:
            print("Y is all 0")
        print_class_distribution(y)
        return X, y
    else:
        print("No data balancing technique applied.")
        return X, y


def experiment_real_labelling(X_train, y_train, X_test, y_ideal, y_real,  c):
    if c == "kmeans":
        model = KMeans(n_clusters=2, random_state=0)
        print("Starting experiment")
        print("Training...")
        start = time()
        model.fit(X_train)
        stop = time()
        print("Testing...")
    elif c == "random_forest":
        model = RandomForestClassifier()
        print("Starting experiment")
        print("Training...")
        start = time()
        model.fit(X_train, y_train)
        stop = time()
        print("Testing...")
    score, y_pred = my_scorer(model, X_test, y_ideal, y_real)
    print("Done.")
    performance = Performance(fit_time=stop-start, precision=score["my_precision"],
                              recall=score["my_recall"], accuracy=score["my_accuracy"],
                              inspection_rate=score["my_inspection_rate"],
                              f1_score=score["my_f1_score"], mcc=score["my_mcc"])
    return performance, y_pred


def print_class_distribution(y):
    print("Dataset Summary: (0 is neutral, 1 is vulnerable)")
    classes = unique(y)
    total = len(y)
    for c in classes:
        n_examples = len(y[y == c])
        percent = n_examples / total * 100
        print('Class %d: %d/%d (%.1f%%)' % (c, n_examples, total, percent))


def get_real_labelling_data(dataset, dataset_releases):
    all_df_path = utils.get_path("my_real_metrics_csv_" + dataset) + "/" + dataset_releases.test_set_release
    all_df_dir = all_df_path
    files = os.listdir(all_df_path)

    num_files = len(files)

    print("Test set: ")
    test_df = pandas.read_csv(os.path.join(all_df_dir, dataset_releases.test_set_release + ".csv"), index_col=0)

    file = Path(files[0]).stem
    train_df = pandas.read_csv(os.path.join(all_df_dir, file + ".csv"), index_col=0)
    selection = 1
    # retrieve training set
    print("Training set: ")
    while selection < num_files:
        file_name = Path(files[selection]).stem
        if file_name != dataset_releases.test_set_release:
            single_df = pandas.read_csv(os.path.join(all_df_dir, file_name + ".csv"), index_col=0)
            train_df = train_df.append(single_df)
        selection = selection + 1
    # data preparation
    train_df.IsIdealVulnerable.replace(('yes', 'no'), (1, 0), inplace=True)
    train_df.dropna(inplace=True)
    train_df.IsRealVulnerable.replace(('yes', 'no'), (1, 0), inplace=True)
    train_df.dropna(inplace=True)
    test_df.IsIdealVulnerable.replace(('yes', 'no'), (1, 0), inplace=True)
    test_df.dropna(inplace=True)
    test_df.IsRealVulnerable.replace(('yes', 'no'), (1, 0), inplace=True)
    test_df.dropna(inplace=True)
    # datasets ready to use
    X_train = train_df.iloc[:, 0:13].values
    #scalerTrain = StandardScaler()
    #X_train = scalerTrain.fit_transform(X_train)
    y_train = train_df.iloc[:, 14].values
    X_test = test_df.iloc[:, 0:13].values
    #scalerTest = StandardScaler()
    #X_test = scalerTest.fit_transform(X_test)
    y_full_test = test_df.iloc[:, 13].values
    y_real_test = test_df.iloc[:, 14].values
    allY = ExperimentResults(test_df.index[0:], y_full_test, y_real_test)

    return X_train, y_train, X_test, y_full_test, y_real_test, allY


def get_timespan_data(dataset, releases, timespan):
    all_df_path = utils.get_path("my_real_metrics_csv_" + dataset) + "/" + releases.test_set_release
    all_df_dir = all_df_path
    print("Test set: ")
    test_df = pandas.read_csv(os.path.join(all_df_dir, releases.test_set_release + ".csv"), index_col=0)

    test_df.IsIdealVulnerable.replace(('yes', 'no'), (1, 0), inplace=True)
    test_df.dropna(inplace=True)
    test_df.IsRealVulnerable.replace(('yes', 'no'), (1, 0), inplace=True)
    test_df.dropna(inplace=True)
    X_test = test_df.iloc[:, 0:13].values
    y_test = test_df.iloc[:, 13].values
    mainbranch_dir = utils.get_path("mainbranch_" + dataset)
    mnb = pyreadr.read_r(os.path.join(mainbranch_dir, "mainbranch_" + dataset + ".Rda"))
    mainbranch = mnb["mainbranch"]
    my_release = releases.test_set_release[8:]
    data_release = mainbranch.loc[my_release, 'dates']
    day_span = datetime.timedelta(days=timespan)
    new_date = data_release + day_span
    my_date = max(mainbranch[mainbranch.dates < new_date].dates)
    my_release = mainbranch.loc[mainbranch.dates == my_date].versions.values[0]
    train_df_path = utils.get_path("my_real_metrics_csv_" + dataset) + "/" + "RELEASE_" + my_release
    train_df_dir = train_df_path
    train_df = pandas.read_csv(os.path.join(train_df_dir, releases.training_set_releases[0] + ".csv"), index_col=0)
    num = len(releases.training_set_releases)

    i = 1
    while i < num:
        single_df = pandas.read_csv(os.path.join(train_df_dir, releases.training_set_releases[i] + ".csv"), index_col=0)
        train_df = train_df.append(single_df)
        i = i+1
    train_df.IsIdealVulnerable.replace(('yes', 'no'), (1, 0), inplace=True)
    train_df.dropna(inplace=True)
    train_df.IsRealVulnerable.replace(('yes', 'no'), (1, 0), inplace=True)
    train_df.dropna(inplace=True)

    X_train = train_df.iloc[:, 0:13].values
    y_train = train_df.iloc[:, 14].values
    y_real = test_df.iloc[:, 14].values
    y_full_test = test_df.iloc[:, 13].values

    return X_train, y_train, X_test, y_test, y_real,


def get_max_date(dataset, releases):
    mainbranch_dir = utils.get_path("mainbranch_" + dataset)
    mnb = pyreadr.read_r(os.path.join(mainbranch_dir, "mainbranch_" + dataset + ".Rda"))
    mainbranch = mnb["mainbranch"]
    max_date = max(mainbranch.dates.values)
    my_release = releases.test_set_release[8:]
    data_release = mainbranch.loc[my_release, 'dates']
    return max_date, data_release


def get_real_labelling_data_manualUp(dataset, dataset_releases):
    all_df_path = utils.get_path("my_real_metrics_csv_" + dataset)+"/" + dataset_releases.test_set_release
    all_df_dir = all_df_path
    files = sorted(Path(all_df_path).iterdir(), key=os.path.getsize)

    num_files = len(files)

    print("Test set: ")
    test_df = pandas.read_csv(os.path.join(all_df_dir, dataset_releases.test_set_release + ".csv"), index_col=0)
    train_df = []

    file = Path(files[0]).stem
    train_df = pandas.read_csv(os.path.join(all_df_dir, file + ".csv"), index_col=0)
    selection = 1
    # retrieve training set
    print("Training set: ")
    while selection < num_files:
        file_name = Path(files[selection]).stem
        if file_name != dataset_releases.test_set_release:
            single_df = pandas.read_csv(os.path.join(all_df_dir, file_name + ".csv"), index_col=0)
            train_df = train_df.append(single_df)
        selection = selection + 1

    train_df.IsIdealVulnerable.replace(('yes', 'no'), (1, 0), inplace=True)
    train_df.dropna(inplace=True)
    train_df.IsRealVulnerable.replace(('yes', 'no'), (1, 0), inplace=True)
    train_df.dropna(inplace=True)
    test_df.IsIdealVulnerable.replace(('yes', 'no'), (1, 0), inplace=True)
    test_df.dropna(inplace=True)
    test_df.IsRealVulnerable.replace(('yes', 'no'), (1, 0), inplace=True)
    test_df.dropna(inplace=True)
    # datasets ready to use
    X_train = train_df.iloc[:, 0:13].values
    scalerTrain = StandardScaler()
    X_train = scalerTrain.fit_transform(X_train)
    y_train = train_df.iloc[:, 13].values
    X_test = test_df.iloc[:, 0:13].values
    scalerTest = StandardScaler()
    X_test = scalerTest.fit_transform(X_test)
    y_full_test = test_df.iloc[:, 13].values
    y_real_test = test_df.iloc[:, 14].values
    allY = ExperimentResults(test_df.index[0:], y_full_test, y_real_test)

    return X_train, y_train, X_test, y_full_test, y_real_test, allY

