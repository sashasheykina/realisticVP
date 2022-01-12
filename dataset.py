import shutil
import pandas
import pyreadr
import os
import requests
import utils
import pandas as pd
import data_preparation

utils.welcome()

app = data_preparation.ask_user_to_choose_project()
output_file = utils.get_path("my_dataset_"+app+"_file")


vulmovement_dir = utils.get_path("vulmovement_" + app)
mainbranch_dir = utils.get_path("mainbranch_" + app)
vuls_dir = utils.get_path("vuls_" + app)
filemetrics_dir = utils.get_path("filemetrics_" + app)
r_metrics_dir = utils.get_path("my_metrics_r_" + app)
output_dir = utils.get_path("my_real_metrics_csv_"+app)


print("Starting task...")
# extract the vulmovement data frame from file
obj = pyreadr.read_r(os.path.join(vulmovement_dir, "vulmovement_" + app + ".Rda"))
vulmovement = obj["vulmovement"]

# extract the mainbranch data frame from file
mnb = pyreadr.read_r(os.path.join(mainbranch_dir, "mainbranch_" + app + ".Rda"))
mainbranch = mnb["mainbranch"]

# extract the vuls data frame from file
vls = pyreadr.read_r(os.path.join(vuls_dir, "vuls_" + app + ".Rda"))
vuls = vls["v"]
count_release = len(vulmovement.columns)
i = 0
while i < count_release:
    j = 0
    release = vulmovement.columns[i]
    i = i + 1
    real_vulnerable_for_training = []
    ideal_vulnerable_for_training = []
    tot_files = []
    try:
        # extract the file metrics dataframe from file
        obj = pyreadr.read_r(os.path.join(r_metrics_dir, "RELEASE_" + release + ".Rda"))
        metrics_df = obj["metrics"]
        output_real_dir = output_dir + "/RELEASE_" + release
        if os.path.exists(output_real_dir):
            shutil.rmtree(output_dir)
        os.mkdir(output_real_dir)
        while j < i:
            version = vulmovement.columns[j]
            j = j+1
            try:
                    # extract the file metrics dataframe from file
                    obj = pyreadr.read_r(os.path.join(r_metrics_dir, "RELEASE_" + version + ".Rda"))
                    metrics_df = obj["metrics"]
                    data_version = mainbranch.loc[version, 'dates']
                    vulnerable_files = []
                    vulnerable_real = []
                    data_release = mainbranch.loc[release, 'dates']
                    if data_version <= data_release:
                        # store vulnerable files names
                        for filename in vulmovement[version]:
                            if filename:
                                subsetDataFrame = vulmovement[vulmovement[version] == filename]
                                hashes = subsetDataFrame.index.values
                                for hash in hashes:
                                    idCVE = vuls.loc[vuls.introhash == hash, 'cveid'].values
                                    size_idcve = len(idCVE)
                                    while size_idcve > 0:
                                        size_idcve = size_idcve - 1
                                        response = requests.get('http://cve.circl.lu/api/cve/' + idCVE[size_idcve])
                                        dataCVE = response.json()["Published"]
                                        date_cve = pd.Timestamp(dataCVE)
                                        if data_release >= date_cve:
                                            vulnerable_real.append(filename)
                                vulnerable_files.append(filename)
                        for real in vulnerable_real:
                            if real not in real_vulnerable_for_training and data_version < data_release:
                                real_vulnerable_for_training.append(real)
                        for ideal in vulnerable_files:
                            if ideal not in ideal_vulnerable_for_training and data_version < data_release:
                                ideal_vulnerable_for_training.append(ideal)
                        try:
                            # extract the file metrics dataframe from file
                            obj = pyreadr.read_r(os.path.join(r_metrics_dir, "RELEASE_" + version + ".Rda"))
                            metrics_df = obj["metrics"]
                            for item in metrics_df.index:
                                if item not in tot_files and data_version < data_release:
                                    tot_files.append(item)
                            # build data frame column to store each file's vulnerability label
                            # first build an all-NEUTRAL column
                            metrics_df["IsIdealVulnerable"] = ["no"] * len(metrics_df.index)

                            metrics_df["IsRealVulnerable"] = ["no"] * len(metrics_df.index)

                            # add VULNERABLE label to rows corresponding to vulnerable files
                            metrics_df.loc[vulnerable_files, ["IsIdealVulnerable"]] = "yes"

                            metrics_df.loc[vulnerable_real, ["IsRealVulnerable"]] = "yes"

                            # save metrics with labels in a new file
                            metrics_df.to_csv(os.path.join(output_real_dir, "RELEASE_" + version + ".csv"))

                        except pyreadr.custom_errors.PyreadrError:
                            # not all versions metrics are stored
                            continue
            except pyreadr.custom_errors.PyreadrError:
                    # not all versions metrics are stored
                continue
    except pyreadr.custom_errors.PyreadrError:
            # not all versions metrics are stored
        continue
    log = data_preparation.Log.build(release, str(len(real_vulnerable_for_training)),
                                     str(len(ideal_vulnerable_for_training)), str(len(tot_files)))
    pandas.DataFrame(log).to_csv(output_file, mode='a', index=False, header=False)
