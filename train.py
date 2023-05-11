# Train a water level QC model

import os
import pandas as pd
import numpy as np
import qc_model_nn as qcmodel 
from optparse import OptionParser
from imblearn.under_sampling import RandomUnderSampler 
from sklearn.utils import shuffle

def main():

  ###########
  # Options #
  ###########
  
  parser = OptionParser()
  parser.add_option("-s", "--stations",
                    default="9751639,8726607",
                    help="List of station IDs (comma-delimited)")
  parser.add_option("-d", "--directory", 
                    default="data/",
                    help="Path to station data directory")
  parser.add_option("-e", "--epochs",
                    default=30,
                    type="int",
                    help="Number of training epochs")
  parser.add_option("-b", "--batch_size",
                    default=256,
                    type="int",
                    help="Batch size")
  parser.add_option(     "--features", 
                    default="PRIMARY,PRIMARY_SIGMA,PRIMARY_SIGMA_TRUE,BACKUP,BACKUP_TRUE,PREDICTION",
                    help="Names of features to select columns for training")
  parser.add_option(     "--no-shuffle",
                    default=False,
                    action="store_true",
                    help="Disable shuffling the test and validation data")
  parser.add_option("-t", "--threshold",
                    default=0.5,
                    type="float",
                    help="Threshold to binarize predictions to class")
  parser.add_option("-r", "--resample_minority_percent", 
                    default=None,
                    type="float",
                    help="Resample data so that N% of the data is the minority class. If `None`, don't resample")

  (options, args) = parser.parse_args()

  # Data directory
  data_dir = options.directory
  if not os.path.exists(data_dir):
    print("Could not find directory {}.\nExiting...".format(data_dir))
  # List of station IDs to include in training
  station_ids = options.stations.split(",")
  # Which columns to include as training features
  featureNames = options.features.split(",")
  # Number of training epochs
  epochs = options.epochs
  # Size of training batch size
  batch_size = options.batch_size
  # Threshold to binarize predictions to classes                
  class_threshold = options.threshold 
  # Resample to achieve N% minority class ("bad data")
  resample_prct = options.resample_minority_percent
  # Shuffle training and validation data
  doShuffle = not options.no_shuffle    


  ################
  # Prepare Data #
  ################

  # Concatenate all stations into single data frame
  data_train_list = [None for station_id in station_ids]
  data_valid_list = [None for station_id in station_ids]

  for i, station_id in enumerate(station_ids):
    # Load the data
    data_train = qcmodel.loadCleanedData(station_id, "train", data_dir)
    targets_train = target_train = data_train.loc[:,['TARGET']]
    data_valid = qcmodel.loadCleanedData(station_id, "validation", data_dir)
    target_valid = data_valid.loc[:,['TARGET']]

    # Resample training data
    if resample_prct is not None:
      rus = RandomUnderSampler(sampling_strategy=resample_prct)
      data_train, targets_train = rus.fit_resample(data_train, targets_train)

    # Add to list
    data_train_list[i] = data_train
    data_valid_list[i] = data_valid

  # Concat 
  data_train = pd.concat(data_train_list)
  data_valid = pd.concat(data_valid_list)
  
  # Shuffle data
  if doShuffle: 
    data_train = shuffle(data_train)
    data_valid = shuffle(data_valid)

  # Select training features
  features_train = data_train.loc[:, featureNames]
  features_valid = data_valid.loc[:, featureNames]

  # Targets
  target_train = data_train.loc[:,['TARGET']]
  target_valid = data_valid.loc[:,['TARGET']]
  

  ###############
  # Train model #
  ###############



  ##############
  # Save model #
  ##############


if __name__ == "__main__":
  main()

