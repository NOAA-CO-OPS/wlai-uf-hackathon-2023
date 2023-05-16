# Train a water level QC model

import os
import pandas as pd
import numpy as np
from optparse import OptionParser
from keras.callbacks import ModelCheckpoint

import qc_model_nn as qcmodel 

def main():

  parser = OptionParser()
  parser.add_option("-s", "--stations",
                    default="9751639,8726607",
                    help="List of station IDs (comma-delimited)")
  parser.add_option("-d", "--directory", 
                    default="data/",
                    help="Path to station data directory")
  parser.add_option("-m", "--model_out", 
                    help="Path to save trained model")
  parser.add_option("-l", "--log_history_out",
                    help="Path to save training history")
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
    print("[-] Could not find directory {}.\nExiting...".format(data_dir))
    exit(-1)
  # Path to save trained model (and checkpoints)
  model_outfile = options.model_out
  if model_outfile is None:
    print("[-] Must provide a path to save model (-m).\nExiting...")
    exit(-1)
  # Path to save training history
  history_outfile = options.log_history_out
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

  # Prepare Datasets
  data_train, features_train, target_train, \
    data_valid, features_valid, target_valid = \
    qcmodel.concat_stations(station_ids, data_dir, featureNames, resample_prct, doShuffle=True)

  # Setup checkpoint
  checkpoint = ModelCheckpoint(model_outfile,
                               monitor="val_loss",
                               verbose=0,
                               save_best_only=True,
                               mode="min")
  callbacks = [checkpoint]
  
  # Initialize model
  numFeatures = len(featureNames)
  model = qcmodel.build(numFeatures)

  # Train 
  history = qcmodel.train(model,
                  features_train,
                  target_train,
                  features_valid,
                  target_valid,
                  epochs=epochs,
                  batch_size=batch_size,
                  callbacks=callbacks)

  if history_outfile is not None:
    history = pd.DataFrame(history.history)
    with open(history_outfile, mode='w') as f:
      history.to_csv(f, index=False)


if __name__ == "__main__":
  main()
