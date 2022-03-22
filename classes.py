from enum import Enum, IntEnum
from datetime import datetime


class Performance:
    def __init__(self, fit_time, precision, recall, accuracy, inspection_rate, f1_score, mcc):
        self.fit_time = fit_time
        self.precision = precision
        self.recall = recall
        self.accuracy = accuracy
        self.inspection_rate = inspection_rate
        self.f1_score = f1_score
        self.mcc = mcc


class DatasetReleases:
    def __init__(self, num_training_set_releases, training_set_releases, test_set_release):
        self.num_training_set_releases = num_training_set_releases
        self.training_set_releases = training_set_releases
        self.test_set_release = test_set_release

    def __str__(self):
        string = "DatasetReleases: ["
        string = string + "num_training_set_releases: " + str(self.num_training_set_releases) + ", "
        string = string + "training_set_releases: ["
        for s in self.training_set_releases:
            string = string + s + " "
        string = string + "], "
        string = string + "test_set_release: " + self.test_set_release + "]"
        return string

    @staticmethod
    def cross_validation():
        return DatasetReleases(-1, ["ALL"], "ALL")


class ExperimentSetting:
    def __init__(self, dataset, approach, validation, balancing, model, run):
        self.dataset = dataset
        self.approach = approach
        self.validation = validation
        self.balancing = balancing
        self.model = model
        self.run = run

    def __str__(self):
        string = "ExperimentSetting: ["
        string = string + "dataset: " + self.dataset + ", "
        string = string + "approach: " + self.approach + ", "
        string = string + "validation: " + self.validation + ", "
        string = string + "balancing: " + self.balancing + ", "
        string = string + "model: " + self.model + ", "
        string = string + "run: " + self.run + "]"
        return string


class ExperimentResults:
    def __init__(self, file, ideal, real):
        self.file = file
        self.ideal = ideal
        self.real = real

    def setPred(self, pred):
        self.pred = pred

    def __str__(self):
        string = "ExperimentResults: ["
        string = string + "dataset: " + self.release + ", "
        string = string + "approach: " + self.ideal + ", "
        string = string + "validation: " + self.real + ", "
        string = string + "balancing: " + self.pred + "]"
        return string


class Log:
    @staticmethod
    def build(experiment_setting, dataset_releases, performance):
        # dd/mm/YY H:M:S
        now_string = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        return {"date_time": [now_string], "dataset": [experiment_setting.dataset],
                "approach": [experiment_setting.approach], "validation": [experiment_setting.validation],
                "balancing": [experiment_setting.balancing], "classifier": [experiment_setting.model],
                "num_training_set_releases": [dataset_releases.num_training_set_releases],
                "test_set_release": [dataset_releases.test_set_release],
                "fit_time": [performance.fit_time], "precision": [performance.precision],
                "recall": [performance.recall], "accuracy": [performance.accuracy],
                "inspection_rate": [performance.inspection_rate], "f1_score": [performance.f1_score],
                "mcc": [performance.mcc]}

    @staticmethod
    def header():
        return {"date_time": [], "dataset": [], "approach": [], "validation": [], "balancing": [], "model": [],
                "num_training_set_releases": [], "test_set_release": [],
                "fit_time": [], "precision": [], "recall": [], "accuracy": [], "inspection_rate": [],
                "f1_score": [], "mcc": []}

    @staticmethod
    def dummy():
        return {"date_time": ["x"], "dataset": ["x"], "approach": ["x"], "validation": ["x"], "balancing": ["x"],
                "model": ["x"], "num_training_set_releases": [0], "test_set_release": ["x"],
                "vocabulary_building_time": [0], "frequency_vectors_building_time": [0], "fit_time": [0],
                "precision": [0], "recall": [0], "accuracy": [0], "inspection_rate": [0], "f1_score": [0], "mcc": [0]}



class LogSpan:
    @staticmethod
    def build(experiment_setting, dataset_releases, performance, timespan, perc_vuln):
        # dd/mm/YY H:M:S
        now_string = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        return {"date_time": [now_string], "dataset": [experiment_setting.dataset],
                "approach": [experiment_setting.approach], "validation": [experiment_setting.validation],
                "balancing": [experiment_setting.balancing], "classifier": [experiment_setting.model],
                "num_training_set_releases": [dataset_releases.num_training_set_releases],
                "test_set_release": [dataset_releases.test_set_release],
                "time_span": [timespan],
                "fit_time": [performance.fit_time], "precision": [performance.precision],
                "recall": [performance.recall], "accuracy": [performance.accuracy],
                "inspection_rate": [performance.inspection_rate], "f1_score": [performance.f1_score],
                "mcc": [performance.mcc], "vulnerability": [perc_vuln]}

    @staticmethod
    def header():
        return {"date_time": [], "dataset": [], "approach": [], "validation": [], "balancing": [], "model": [],
                "num_training_set_releases": [], "test_set_release": [], "trimester": [],
                "fit_time": [], "precision": [], "recall": [], "accuracy": [], "inspection_rate": [],
                "f1_score": [], "mcc": [], "%vulnerability": []}

    @staticmethod
    def dummy():
        return {"date_time": ["x"], "dataset": ["x"], "approach": ["x"], "validation": ["x"], "balancing": ["x"],
                "model": ["x"], "num_training_set_releases": [0], "test_set_release": ["x"], "fit_time": [0],
                "time_span": [0], "precision": [0], "recall": [0], "accuracy": [0], "inspection_rate": [0], "f1_score": [0], "mcc": [0]}


class Run(IntEnum):
    run_real = 1
    run_timespan = 2


class Dataset(IntEnum):
    phpmyadmin = 1
    moodle = 2


class Validation(IntEnum):
    real_labelling = 1


class Model(IntEnum):
    kmeans = 1
    random_forest = 2


class Approach(IntEnum):
    metrics = 1


class Balancing(IntEnum):
    none = 1
    undersampling = 2
    oversampling = 3
