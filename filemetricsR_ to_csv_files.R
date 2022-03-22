print("INFO SUL PROGRAMMA")

#clean the environment
rm(list = ls())

path_config = read.csv("path_config.csv", header = TRUE, row.names = 1)

print("Please choose app")
app_name = switch(menu(c("Moodle", "PHPMyAdmin")), "moodle", "phpmyadmin")

1
input_dir_key = paste("filemetrics_", app_name, sep = "")
input_file_name = input_dir_key

#load metrics
input_file = paste(path_config[input_dir_key, "path"], "/", input_file_name, ".Rda", sep = "")
load(file = input_file)

print("Exporting metrics to directory:")
output_dir_key = paste("my_filemetrics_r_", app_name, sep = "")
output_dir = path_config[output_dir_key, "path"]
print(output_dir)

task = function() {
  print("Starting task...")
  #save metrics of each release in a separate file
  for (release_name in names(filemetrics)) {
    single_file_name = paste(output_dir, "/", release_name, ".Rda", sep = "")
    metrics = filemetrics[[release_name]]
    save(metrics, file = single_file_name)
  }
  print("Done.")
}

if (dir.exists(output_dir)) {
  print("Directory already exists: maybe the task has alredy been executed.")
  print("Do you want to continue? (Duplicate files will be overwritten)")
  switch(menu(c("Yes, continue", "No, exit program")), task(), {})
} else {
  dir.create(output_dir)
  task()
}

#clean the environment
rm(list = ls())

print("Bye!")
