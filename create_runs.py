# Generate a shell script where each line is a training or evaluation run

import numpy as np
import pandas as pd
from optparse import OptionParser
import itertools

def create_command(station_ids, name, data_dir, output_dir, hyperparams, trial=1):
  # Returns a string containing a python command to train the model
  # Options:
  #   station_ids: list of station IDs
  #   name: arbitrary name to be added to the output files for identification
  #   data_dir: path to directory with station data
  #   output_dir: path to directory to save output files
  #   hyperparams: string containing additional hyperparams (e.g. '-e 10 -r 0.1')

  cmd_str = "python train.py " + \
            " -s " + str(station_ids) + \
            " -m " + output_dir + "/model_" + str(name) + "_trial-" + str(trial) + ".hdf5" + \
            " -l " + output_dir + "/trainlog_" + str(name)  + "_trial-" + str(trial) + ".csv" + \
            "  " + hyperparams
  return cmd_str


parser = OptionParser()
parser.add_option("-s", "--stations_file",
                  default="stations.csv",
                  help="Path to stations table.")
parser.add_option("-c", "--columns",
                  default="station_id",
                  help="List of column names for grouping stations (comma-delimited). Or, 'ALL' to combine all stations.")
parser.add_option("-d", "--data_dir",
                  default="data/",
                  help="Path to station data directory.")
parser.add_option("-o", "--output_dir",
                  default="out/",
                  help="Path to output directory.")
parser.add_option("-e", "--epochs",
                  help="Number of training epochs.")
parser.add_option("-b", "--batch_size",
                  help="Batch size")
parser.add_option("-r", "--resample_minority_percent",
                  help="Resample data so that N% of the data is the minority class. If `None`, don't resample.")
parser.add_option(      "--checkpoint",
                  default="False",
                  action="store_true",
                  help="Whether or not to use model checkpoints.")
parser.add_option("-t", "--trials",
                  default=1,
                  type="int",
                  help="Number of trials for each set of hyperparameters.")
(options, args) = parser.parse_args()

stations_file = options.stations_file
group_columns = options.columns.split(",")
output_dir = options.output_dir
data_dir = options.data_dir
num_trials = options.trials

# Create a dictionary of hyperparamers
# Only add hyperparams that are provided (otherwise, will use default values in `train.py`)
hyperparams = {}
if options.epochs is not None:
  hyperparams["-e"] = np.array(options.epochs.split(",")).astype("int")
if options.batch_size is not None:
  hyperparams["-b"] = np.array(options.batch_size.split(",")).astype("int")
if options.resample_minority_percent is not None:
  hyperparams["-r"] = np.array(options.resample_minority_percent.split(",")).astype("float")
if options.checkpoint is not None:
  hyperparams["--checkpoint"] = np.array([""])

# Load stations table
dfStations = pd.read_csv(stations_file)
# Get list of unique station IDs
station_ids = np.unique(dfStations["station_id"].astype("string"))

# Build list of strings for all combinations of hyperparams to sweep through
hyperparam_combos = [[hpk + " " + str(v) for v in hyperparams[hpk]] for hpk in hyperparams.keys()]
hyperparam_combos = list(itertools.product(*hyperparam_combos))
hyperparam_combos = [" ".join(hpc) for hpc in hyperparam_combos]

# For each column name provided, combine the stations and print the training run commands
for group_column in group_columns:

  # Special case: column name 'ALL' is not a column. Instead, use all stations.
  if group_column == "ALL":
    # Use all station IDs
    station_ids_str = ",".join(station_ids)
    # Sweep over hyperparameters for all stations
    for hyperparam_combo in hyperparam_combos:
      # Repeat for each trial
      for i in range(1, num_trials+1):
        cmd_str = create_command(station_ids_str, "all-stations", data_dir, output_dir, hyperparam_combo, trial=i)
        # Print the training Python command
        print(cmd_str)
    # Continue to next column name
    continue

  # Normal case: use the column name to group stations IDs by that column value (e.g. into geographic regions)
  try:
    # Get all the values within that column (to be treated as station groups)
    groups = np.unique(dfStations[group_column])
  except KeyError:
    print("Could not find column '{}' in stations file '{}'\nExiting...".format(group_column, stations_file))
    exit(-2)

  # Combine the stations within each group
  for group in groups:
    # Subset the stations by group 
    dfStationsGroup = dfStations[dfStations[group_column] == group]
    group_station_ids = np.unique(dfStationsGroup["station_id"].astype("string"))
    group_station_ids_str = ",".join(group_station_ids)
    # Sweep over hyperparameters for stations in this group
    for hyperparam_combo in hyperparam_combos:
      # Repeat for each trial
      for i in range(1, num_trials+1):
        cmd_str = create_command(group_station_ids_str, group, data_dir, output_dir, hyperparam_combo, trial=i)
        # Print the training Python command
        print(cmd_str)
