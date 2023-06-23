#Software Vulnerability Prediction based on supervised vs unsupervised learning for Real-World Labelling

## Requirement

- Python 3.8
- pip3

##Run

1. cd realunsuplearning
2. python3 run.py					#to generate single experiment
3. python3 run_all_experiments.py 	#to generate all experiments  

## Project Structure

- DATASET
  - Original source of the dataset
- PROJECT DATA
  - **Empty directory**, generated dataset will be here
- REPORT
  - **Empty directory**, generated report will be here
- Source code

## Module Description

- [run.py](realunsuplearning/run.py)
  - Entry module that trigger everyother modules to start the process (single experiment)
- [run_all_experiments.py](realunsuplearning/run_all_experiments.py)
  - Entry module that trigger everyother modules to start the process (all experiments)
- [dataset.py](realunsuplearning/dataset.py)
  - Handling dataset preprocessing
- [implementation.py](realunsuplearning/implementation.py)
  - Contain functions that take feature and label as input and return the data with selected metrics depending on different implementations
- [classes.py](realunsuplearning/classes.py)
  - Contains all the framework variables ( e.g. Dataset, Performance, Cluster, Approach, ExperimentResults ecc. )
- [experiments.py](realunsuplearning/experiments.py)
  - Contains functions that setting all variables for experiments
- [dataset_moodle.csv, dataset_phpmyadmin.csv](realunsuplearning/dataset_moodle.csv, realunsuplearning/dataset_phpmyadmin.csv )
  - Dataset ad hoc that contains Ideal-World-Labelling, Real-World-Labelling and Total number of files for each release  
- [experiments.csv, experiments_kmeans_cmeans.csv, experiment_with_StandarsScaler.csv ecc](realunsuplearning/.. )
  - Generated in the second phase of the experiment and contains Performance Evaluation (Precision, Accuracy, Mcc ecc.)

## Default dataset link

- [MOODLE, PHPMYADMIN]
	- {File metrics for PHPMyAdmin, File metrics for Moodle, Vulnerability data for PHPMyAdmin, Vulnerability data for Moodle, Vulnerability tracking data for PHPMyAdmin,
		Vulnerability tracking data for Moodle, Main branch release metadata for PHPMyAdmin, Main branch release metadata for Moodle}: (https://seam.cs.umd.edu/webvuldata/data.html)
