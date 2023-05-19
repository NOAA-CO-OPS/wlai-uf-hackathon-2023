# Generate a shell script where each line is a training or evaluation run

import numpy as np
import pandas as pd
from optparse import OptionParser
import itertools

def create_command(station_ids, name, data_dir, output_dir, hyperparams):
  cmd_str = "python train.py " + \
            " -s " + str(station_ids) + \
            " -m " + "model_" + str(name) + ".hdf5" + \
            " -l " + "trainlog_" + str(name) + ".csv" + \
            "  " + hyperparams
  return cmd_str


parser = OptionParser()
parser.add_option("-s", "--stations_file",
                  default="stations.csv",
                  help="Path to stations table.")
parser.add_option("-c", "--columns",
                  default="station_id",
                  help="List of column names for grouping stations (comma-delimited).")
parser.add_option("-d", "--data_dir",
                  default="data/",
                  help="Path to station data directory.")
parser.add_option("-o", "--output_dir",
                  default="out/",
                  help="Path to output directory.")
parser.add_option("-e", "--epochs",
                  help="Number of training epochs")
parser.add_option("-b", "--batch_size",
                  help="Batch size")
parser.add_option("-r", "--resample_minority_percent",
                  help="Resample data so that N% of the data is the minority class. If `None`, don't resample")
(options, args) = parser.parse_args()

stations_file = options.stations_file
group_columns = options.columns.split(",")
output_dir = options.output_dir
data_dir = options.data_dir

hyperparams = {}
if options.epochs is not None:
  hyperparams["-e"] = np.array(options.epochs.split(",")).astype("int")
if options.batch_size is not None:
  hyperparams["-b"] = np.array(options.batch_size.split(",")).astype("int")
if options.resample_minority_percent is not None:
  hyperparams["-r"] = np.array(options.resample_minority_percent.split(",")).astype("float")

dfStations = pd.read_csv(stations_file)
station_ids = dfStations["station_id"].astype("string")

hyperparam_combos = [[hpk + " " + str(v) for v in hyperparams[hpk]] for hpk in hyperparams.keys()]
hyperparam_combos = list(itertools.product(*hyperparam_combos))
hyperparam_combos = [" ".join(hpc) for hpc in hyperparam_combos]

for group_column in group_columns:

  groups = dfStations[group_column]
  for group in groups:
    dfStationsGroup = dfStations[dfStations[group_column] == group]
    group_station_ids = np.unique(dfStationsGroup["station_id"].astype("string"))
    group_station_ids_str = ",".join(group_station_ids)

    for hyperparam_combo in hyperparam_combos:
      cmd_str = create_command(group_station_ids_str, group, data_dir, output_dir, hyperparam_combo)

      print(cmd_str)
