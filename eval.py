# Evaluate a trained water level QC model

import os
import pandas as pd
import numpy as np
from optparse import OptionParser
from keras.models import load_model
from sklearn.metrics import roc_auc_score as roc
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools
import qc_model_nn as qcmodel

def plotConfusionMatrix(confusion_matrix, classes,
                        normalize=False,
                        title="Confusion matrix",
                        cmap=plt.cm.Blues):
  """
  This function prints and plots the confusion matrix.
  Normalization can be applied by setting `normalize=True`. 
  """
  cm = confusion_matrix
  
  if normalize:
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
  
  plt.imshow(cm, interpolation='nearest', cmap=cmap)
  plt.title(title)
  fmt = '.3f' if normalize else 'd'
  thresh = cm.max() / 2.
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, format(cm[i, j], fmt),
             horizontalalignment="center",
             verticalalignment='center',
             color="white" if cm[i, j] > thresh else "black")

  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  xtick_marks = np.array([0,1])
  plt.xticks(xtick_marks, classes)
  ytick_marks=np.array([-.5,0,1,1.5])
  ylabels=['',classes[0],classes[1],'']
  plt.yticks(ytick_marks, ylabels,rotation=0)
  plt.colorbar()
  plt.tight_layout()
  return


def calc_metrics(probs, targets, threshold=0.5):
  # Convert to numpy arrays
  probs = np.array(probs)
  targets = np.array(targets)
  preds = 1 * (probs >= threshold)
  
  # Number of points
  n = len(preds)
 
  # Calculate metrics
  metrics = {}
  # Confusion matrix
  cnfMatrix = confusion_matrix(targets, preds)
  cnfMatrix_ = cnfMatrix.ravel()
  metrics["hits"] = cnfMatrix_[0]
  metrics["misses"] = cnfMatrix_[2]
  metrics["false_alarms"] = cnfMatrix_[1]
  metrics["correct_rejects"] = cnfMatrix_[3]
  # Number of points
  metrics["num_points"] = n
  # Number of good points
  metrics["num_good_targets"] = len(targets[targets == 1])
  # Propoertion of good points
  metrics["prop_good_targets"] = metrics["num_good_targets"] / n
  # Number of bad points
  metrics["num_bad_targets"] = len(targets[targets == 0])
  # Proportion of bad points
  metrics["prop_bad_targets"] = metrics["num_bad_targets"] / n
  # Accuracy
  metrics["accuracy"] = 1 - np.sum(np.abs(preds - targets)) / n
  # Bad point class accuracy 
  idxs = np.where(targets == 0)
  metrics["accuracy_bad_points"] = 1 - np.sum(np.abs(preds[idxs] - targets[idxs])) / n
  # Area under the ROC
  metrics["area_under_roc"] = roc(targets, probs)

  return metrics, cnfMatrix


def main():

  parser = OptionParser()
  parser.add_option("-p", "--output_prefix", 
                    help="File prefix for output files")
  parser.add_option("-m", "--model",
                    help="Path to trained model")
  parser.add_option("-s", "--stations",
                    default="9751639,8726607",
                    help="List of station IDs (comma-delimited)")
  parser.add_option("-d", "--directory",
                    default="data/"
                    help="Path to station data directory")
  parser.add_option(     "--features",
                    default="PRIMARY,PRIMARY_SIGMA,PRIMARY_SIGMA_TRUE,BACKUP,BACKUP_TRUE,PREDICTION",
                    help="Names of features to select columns for training")
  parser.add_option("-t", "--threshold",
                    default=0.5,
                    type="float",
                    help="Threshold to binarize predictions to class")
  parser.add_option("-l", "--log_history",
                    help="Path to training curve data")
  (options, args) = parser.parse_args()

  # File prefix
  outfile_prefix = options.output_prefix
  if outfile_prefix is None:
    print("[-] Must provide an output file prefix for saving output files (-p).\nExiting...")
    exit(-1)
  # Path to trained model
  model_file = options.model
  if model_file is None:
    print("[-] Must provide a path to a trained model (-m).\nExiting...")
    exit(-1)
  if not os.path.exists(model_file):
    print("[-] Could not find trained model file {}.\nExiting...".format(model_file))
  # Data directory
  data_dir = options.directory
  if not os.path.exists(data_dir):
    print("[-] Could not find directory {}.\nExiting...".format(data_dir))
    exit(-1)
  # List of station IDs to include in training
  station_ids = options.stations.split(",")
  # Which columns to include as training features
  featureNames = options.features.split(",")
  # Threshold to binarize predictions to classes
  class_threshold = options.threshold
  # Path to training curve history
  training_history_file = options.log_history

  # Prepare data
  data_train, features_train, targets_train, \
    data_valid, features_valid, targets_valid = \
    qcmodel.concat_stations(station_ids, data_dir, featureNames)

  numTrain = len(data_train)
  numValid = len(data_valid)
 
  # Load model
  model = load_model(model_file)

  # Predictions
  probs_train = np.array(model.predict(features_train))[:,0]
  preds_train = 1 * (probs_train >= class_threshold)
  probs_valid = np.array(model.predict(features_valid))[:,0]
  preds_valid = 1 * (probs_valid >= class_threshold)
 
  targets_train = np.array(targets_train)[:,0]
  targets_valid = np.array(targets_valid)[:,0]

  def outcomes2df(probs, preds, targets):
    df = pd.DataFrame({
      "probability"            : probs, 
      "predicted_class"        : preds,
      "predicted_class_label"  : ["bad" if x == 0 else "good" for x in preds],
      "target_class"           : targets,
      "predicted_target_label" : ["bad" if x == 0 else "good" for x in targets],
    })
    return df

  dfTrain = outcomes2df(probs_train, preds_train, targets_train)
  dfValid = outcomes2df(probs_valid, preds_valid, targets_valid)

  # Calculate metrics
  train_metrics, train_confusionMatrix = \
    calc_metrics(probs_train, targets_train, class_threshold)
  valid_metrics, valid_confusionMatrix = \
    calc_metrics(probs_valid, targets_valid, class_threshold)
  # Init Metrics table
  dfMetrics = pd.DataFrame(
    columns=["stations", "dataset", "num_points",
             "hits", "misses", "false_alarms", "correct_rejects",
             "num_good_targets", "prop_good_targets",
             "num_bad_targets", "prop_good_targets",
             "accuracy", "accuracy_bad_points", "area_under_roc"])
  dfMetrics["stations"] = [options.stations, options.stations]
  dfMetrics["dataset"] = ["train", "validate"]
  # Add metrics to table
  def populateMetrics(dfMetrics, metrics):
    for key in metrics.keys():
      dfMetrics[key] = metrics[key]
  populateMetrics(dfMetrics, train_metrics)
  populateMetrics(dfMetrics, valid_metrics)

  # Outputs

  # Model outputs (table)
  dfTrain_outfile = outfile_prefix + "-output-train.csv"
  dfTrain.to_csv(dfTrain_outfile, index=False)

  dfValid_outfile = outfile_prefix + "-output-validate.csv"
  dfValid.to_csv(dfValid_outfile, index=False)

  # Metrics (table)
  dfMetrics_outfile = outfile_prefix + "-metrics.csv"
  dfMetrics.to_csv(dfMetrics_outfile, index=False)

  # Training curve (graphic)
  training_curve_outfile = outfile_prefix + "-training_curve.pdf"
  if training_history_file is not None:
    dfHistory = pd.read_csv(training_history_file)
    fig, ax = plt.subplots()
    ax.plot(dfHistory['loss'])
    ax.plot(dfHistory['val_loss'])
    ax.legend(['Training','Validation'])
    fig.savefig(training_curve_outfile)

  # Confusion matrix (graphic)
  plt.clf()
  confusion_matrix_outfile_train = outfile_prefix + "-confusionmatrix-train.pdf"
  classNames = ['bad data point','good data point']
  plotConfusionMatrix(train_confusionMatrix, classes=classNames,
                      normalize=True, title='Confusion matrix, with normalization')
  fig.savefig(confusion_matrix_outfile_train)
  plt.clf()
  confusion_matrix_outfile_valid = outfile_prefix + "-confusionmatrix-validate.pdf"
  plotConfusionMatrix(valid_confusionMatrix, classes=classNames,
                      normalize=True, title='Confusion matrix, with normalization')
  fig.savefig(confusion_matrix_outfile_valid)


if __name__ == "__main__":
  main()
