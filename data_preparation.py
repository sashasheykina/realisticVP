import utils
import os
import shutil
import sys


def ask_user_to_choose_project():
    while True:
        print("Please choose app")
        print("1: Moodle")
        print("2: PHPMyAdmin")
        selection = input("Selection:")
        if selection == "1":
            print("Working on app: Moodle")
            return "moodle"
        elif selection == "2":
            print("Working on app: PHPMyAdmin")
            return "phpmyadmin"
        else:
            print("Invalid selection!")


def check_output_directory(output_dir):
    if os.path.exists(output_dir):
        print("Directory " + output_dir + " already exists.")
        print("Maybe the task has already been executed.")
        print("Do you want to continue anyway? (Directory will be overwritten)")
        print("y: Yes, continue with task execution")
        print("any: No, exit the program")
        selection = input("Selection:")

        if (selection != "y") & (selection != "Y"):
            utils.bye()
        else:
            shutil.rmtree(output_dir)
            os.mkdir(output_dir)
    else:
        try:
            os.mkdir(output_dir)
        except OSError:
            print("Creation of output directory %s failed" % output_dir)
            sys.exit()
        else:
            print("Successfully created output directory %s " % output_dir)


class Log:
    @staticmethod
    def build(test_release, real_vulnerable_file, ideal_vulnerable_file, tot_files):
        return {"test_release": [test_release],
                "#real_vulnerable_file_for_traing_set": [real_vulnerable_file],
                "ideal_vulnerable_file_for_traing_set": [ideal_vulnerable_file],
                "#tot_files_for_traing_set": [tot_files]}

    @staticmethod
    def header():
        return {"test_release": [], "#real_vulnerable_file": [], "#ideal_vulnerable_file": [], "#tot_files": []}
