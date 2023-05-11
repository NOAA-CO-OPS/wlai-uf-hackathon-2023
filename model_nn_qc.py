# Functions to run and evaluate the Neural Network model

import os
import pandas as pd
from optparse import OptionParser
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras import regularizers
from keras.models import load_model
from keras.callbacks import ModelCheckpoint


def build(numFeatures, 
          dense_nodes=[32, 32],
          activation_fn='relu',
          dropout=0.25):
  
  # Define architecture
  model = Sequential()
  for dense_n in dense_nodes:
    model.add(Dense(dense_n, activation=activation_fn, input_dim=numFeatures))
    model.add(Dropout(dropout))
  model.add(Dense(1, activation='sigmoid'))

  # Compile model
  model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])

  # Print summary
  print(model.summary())

  return model


def train(model, dataTrain, targetTrain, dataVal, targetVal, 
          epochs=10, batch_size=32, callbacks=None):
  history = model.fit(dataTrain, 
                      targetTrain,
                      epochs=epochs,
                      batch_size=batch_size,
                      validation_data=(dataVal, targetVal),
                      callbacks=callbacks)
  return history


def loadCleanedData(stationNum, fileType, dataDirectory):
  
  # Where stationNum is the wl station number to load
  # filetype is 'test','train' or 'validation'
  # dataDirectory is the directory where the test, train, validation sub-directories are located

  columns = [
    'STATION_ID',
    'DATE_TIME', 
    'SENSOR_USED_PRIMARY', 
    'PRIMARY', 
    'PRIMARY_TRUE', 
    'PRIMARY_SIGMA',
    'PRIMARY_SIGMA_TRUE',
    'PRIMARY_RESIDUAL',
    'BACKUP',
    'BACKUP_TRUE',
    'BACKUP_SIGMA',
    'BACKUP_SIGMA_TRUE',
    'BACKUP_RESIDUAL',
    'PREDICTION',
    'VERIFIED',
    'TARGET',
    'NEIGHBOR_PRIMARY',
    'NEIGHBOR_PREDICTION',
    'NEIGHBOR_PRIMARY_RESIDUAL',
    'NEIGHBOR_TARGET'
  ]

  filenameIn = dataDirectory + '/' + stationNum + '_processed_ver_merged_wl_' + fileType + '.csv'

  dataIn = pd.read_csv(filenameIn, 
                       index_col=1, 
                       parse_dates=True,
                       usecols=columns)
  dataIn.index.name ='time'

  # Remove all cases where PRIMARY_TRUE = 0, since then the primary data is missing, and we aren't assessing its quality
  dataIn.drop(dataIn[dataIn['PRIMARY_TRUE'] == 0].index, inplace = True) 

  return dataIn


if __name__ == "__main__":

  parser = OptionParser()
  parser.add_option("-s", "--station", 
                    default="9751639",
                    help="Water level station ID")
  parser.add_option("-d", "--directory", 
                    default="data/",
                    help="Path to station data")
  parser.add_option("-e", "--epochs",
                    default=5,
                    type="int",
                    help="Number of training epochs")
  parser.add_option("-b", "--batch_size",
                    default=256,
                    type="int",
                    help="Batch size")
  (options, args) = parser.parse_args()

  station_id = options.station
  data_dir = options.directory
  epochs = options.epochs
  batch_size = options.batch_size
  
  # Example options
  featureNames = ['PRIMARY',
                  'PRIMARY_SIGMA',
                  'PRIMARY_SIGMA_TRUE',
                  'PRIMARY_RESIDUAL',
                  'BACKUP',
                  'BACKUP_TRUE',
                  'PREDICTION'
                  ]
  # Training data
  data_train = loadCleanedData(station_id, "train", data_dir)
  target_train = data_train.loc[:,['TARGET']]
  data_train = data_train.loc[:, featureNames]

  # Validation data
  data_validate = loadCleanedData(station_id, "validation", data_dir)
  target_validate = data_validate.loc[:,['TARGET']]
  data_validate = data_validate.loc[:, featureNames]

  # Build model
  numFeatures = len(featureNames)
  model = build(numFeatures)

  # Train model
  history = train(model, 
                  data_train, 
                  target_train,
                  data_validate,
                  target_validate,
                  epochs=epochs, 
                  batch_size=batch_size, 
                  callbacks=None)

  # Evaluate
  # Get validation predictions
  probs = model.predict(data_validate)
  # Binarize predictions to class
  preds = 1 * (probs >= 0.5)
  # Compute confusion matrix
  cnfMatrix = confusion_matrix(target_validate, preds)
  # Format for print
  cmtx = pd.DataFrame(cnfMatrix, 
                      index=['true: bad data', 'true: good data'], 
                      columns=['pred: bad data', 'pred: good data'])
  print("\nConfusion Matrix for Station {}:\n".format(station_id))
  print(cmtx)
