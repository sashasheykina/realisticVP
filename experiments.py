from classes import Dataset, Validation, Approach, Balancing, DatasetReleases, ExperimentSetting, Model, Run
import utils
import os
from natsort import natsorted


def choose(enumeration):
    utils.print_space()
    print("Please choose " + enumeration.__name__)
    items = list(enumeration)
    for x in items:
        print(str(x.value) + ": " + x.name)
    selection = input("Selection:")
    selection = selection.strip()
    if selection.isnumeric():
        if int(selection) <= len(items):
            chosen = items[int(selection)-1].name
            print("Selected: " + chosen)
            return chosen
    print("Invalid selection!")
    return choose(enumeration)


def set_experiment(i):
    utils.print_space()
    print("Set experiment " + str(i))
    run = "run_timespan"#choose(Run)
    dataset = "phpmyadmin"#choose(Dataset)
    model = "random_forest"#choose(Model)
    validation = choose(Validation)
    approach = "metrics"#choose(Approach)
    if validation == "real_labelling":
        releases = choose_releases(dataset)
    balancing = "oversampling"#choose(Balancing)
    setting = ExperimentSetting(dataset, approach, validation, balancing, model, run)
    utils.print_space()
    return setting, releases


def choose_releases(dataset):
    utils.print_space()

    training_set_releases = []

    # select test release
    all_df_dir = utils.get_path("my_metrics_csv_" + dataset)

    all_df_file_names = os.listdir(all_df_dir)

    all_df_file_names = natsorted(all_df_file_names)
    num_files = len(all_df_file_names)
    start_list_from = 0
    print("Please choose test release")
    for i in range(num_files):
        if i >= start_list_from:
            print(str(i) + ": " + all_df_file_names[i][:-4])
    loop = True
    while loop:
        selection = input("Selection: ")
        if selection.isnumeric():
            test_set_release_index = int(selection)
            test_set_release = all_df_file_names[int(selection)][:-4]
            print("Test release: "+test_set_release)
            loop = False
        else:
            print("Invalid selection!")

    # retrieve training releases
    for i in range(test_set_release_index):
        training_set_releases.append(all_df_file_names[test_set_release_index-i-1][:-4])
    print("Training releases: "+str(training_set_releases))

    return DatasetReleases(test_set_release_index, training_set_releases, test_set_release)


def want_further_experiment():
    utils.print_space()
    print("Do you want to set another experiment?")
    print("1: Yes, set another experiment")
    print("2: No, start running the experiments")
    selection = input("Selection:")
    selection = selection.strip()
    if selection == "1":
        return True
    elif selection == "2":
        return False
    else:
        print("Invalid selection!")
        return want_further_experiment()


def generate_all_experiments_settings():
    all_experiments_list = []
    datasets = ["moodle"]
    approaches = ["metrics"]
    balancing = ["none", "undersampling", "oversampling"]
    cluster = ["kmeans", "random_forest"]

    # release-based
    for d in datasets:
        for a in approaches:
            for b in balancing:
                for c in cluster:
                    setting = ExperimentSetting(d, a, "real_labelling", b, c, "run_real")
                    all_releases = generate_all_releases(d)
                    for releases in all_releases:
                        all_experiments_list.append((setting, releases))
    return all_experiments_list


def generate_all_releases(dataset):
    utils.print_space()

    all_releases = []

    # select test release
    all_df_dir = utils.get_path("my_metrics_csv_" + dataset)

    all_df_file_names = os.listdir(all_df_dir)

    all_df_file_names = natsorted(all_df_file_names)
    num_files = len(all_df_file_names)
    print("numero file nel dataset: "+dataset + " "+str(num_files))

    selection = 3
    while selection < num_files-1:
        training_set_releases = []
        selection = selection + 1
        test_set_release_index = int(selection)
        test_set_release = all_df_file_names[int(selection)][:-4]
        print("Test release: " + test_set_release)

        # retrieve training releases
        for i in range(test_set_release_index):
            training_set_releases.append(all_df_file_names[test_set_release_index - i - 1][:-4])
        print("Training releases: " + str(training_set_releases))

        all_releases.append(DatasetReleases(test_set_release_index, training_set_releases, test_set_release))
    return all_releases


