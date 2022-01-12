import utils
from experiments import set_experiment, want_further_experiment, generate_all_experiments_settings
from classes import Log
from implementation import execute_experiment
import pandas
import os

utils.welcome()

output_file = utils.get_path("my_experiments_csv_file")
if not os.path.exists(output_file):
    # prepare csv file for experiments output
    header = Log.header()
    pandas.DataFrame(header).to_csv(output_file, index=False)

experiments_to_run = []
all_experiments = generate_all_experiments_settings()
n = len(all_experiments)
print(n)
print(all_experiments[0][0].approach)
print(all_experiments[0][1])

for i in range(0, n):
    log = execute_experiment(i+1, all_experiments[i])
    pandas.DataFrame(log).to_csv(output_file, mode='a', index=False, header=False)

utils.print_space()
print("All done!")
print(str(n) + " experiments saved to file: " + output_file)
utils.bye()
