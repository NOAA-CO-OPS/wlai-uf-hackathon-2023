# Read & format water level QC data

import os
import pandas as pd
import numpy as np
from optparse import OptionParser

import qc_model_nn as qcmodel

def main():

  parser = OptionParser()
  parser.add_option("-s", "--stations",
                    default="9751639,8726607",
                    help="List of station IDs (comma-delimited).")
  parser.add_option("-d", "--directory",
                    default="data/",
                    help="Path to station data directory")
  parser.add_option("-o", "--output_csv",
                    help="Path to save output table (.csv).")
  parser.add_option("-c", "--choice",
                    default="train",
                    help="Select train, validate, or test.")
  (options, args) = parser.parse_args()

  # List of station IDs to include
  station_ids = options.stations.split(",")
  
  # Selection of data type
  choice = options.choice
  choices = ["train", "validate", "test"]
  if choice not in choices:
    print("Expected data selection from the following choices: {}".format(
      choices))

  # Data directory
  data_dir = options.directory
  if not os.path.exists(data_dir):
    print("[-] Could not find directory {}.\nExiting...".format(data_dir))
    exit(-1)

  # Output csv
  outfile = options.output_csv

  data_train, _, target_train, \
    data_valid, _, target_valid = \
    qcmodel.concat_stations(station_ids, data_dir, None, None, doShuffle=False)
  
  dfs = {
    "train" : data_train,
    "validate" : data_valid,
  }

  dfs[choice].to_csv(outfile, index=False)

if __name__ == "__main__":
  main()
