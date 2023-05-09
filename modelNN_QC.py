# Greg's finalized NeuralNetwork (NN) model for classifying good and bad data points (QC) 
# ________________________________________________________________________________________________________________

# Packages needed including modelNN_functions
# ________________________________________________________________________________________________________________
import os
from pathlib import Path
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import itertools
import dill
import pickle
from scipy import io
from scipy.signal import detrend as detrend
from sklearn.utils import shuffle
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.metrics import brier_score_loss
from sklearn.metrics import roc_auc_score as roc
from sklearn.metrics import roc_curve
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras import regularizers
from keras.models import load_model
from keras.callbacks import ModelCheckpoint

# Functions from modelNN_functions

from modelNN_functions import assessTrainTestData
from modelNN_functions import plotConfusionMatrix
from modelNN_functions import pandasToMat
from modelNN_functions import BSS
from modelNN_functions import loadCleanedData
from modelNN_functions import resampleGoodPointsSetNum
# ________________________________________________________________________________________________________________

# Define the NN model to be used for the QC (good or bad 1=good, 0=bad)

def runNNmodel(featureTrain, targetTrain, featureTest, targetTest, predictFeatures, predictTargets, 
               predictStations, epochCount, batchCount, classThreshold, numFeatures):

    """
    where: 
        featureTrain = features used in training the model
        targetTrain = targets (0 or 1) used in training the model
        featureTest = features used for validation
        targetTest = targets (0 or 1) used for valdation
        predictFeatures = the features for the model predictions
        predictTargets = the targets for the model predictions
        predictStations = the stations for the model predictions
        epochCount = number of training cycles to use for NN model (10-20 seems reasonable)
        batchCount = the batch size to use when training (somewhere from 32-256 seems reasonable)
        classThreshold = the threshold from 0 to 1 used to classify a point as good or bad (default is 0.5)
        numFeatures = the number of input features in the NN model
    """

    # First define the keras NN framework to be used
    # For a single-input model with 2 classes (binary classification):
    model = Sequential()
    #model.add(Dense(16, kernel_regularizer=regularizers.l2(0.001), activation='relu', input_dim=numFeatures))
    model.add(Dense(32, activation='relu', input_dim=numFeatures))
    model.add(Dropout(0.25))
    #model.add(Dense(16, kernel_regularizer=regularizers.l2(0.001), activation='relu'))
    model.add(Dense(32, activation='relu', input_dim=numFeatures))
    model.add(Dropout(0.25))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    
    # Set up the model checkpoint
    # checkpoint
    filepath="model_best.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]
    
    # Train the model, iterating on the data in batches of X # samples (somewhere between 32 - 256)
    history = model.fit(featureTrain, targetTrain, epochs=epochCount, batch_size=batchCount, validation_data=(featureTest, targetTest), callbacks=callbacks_list)
    
    # Load the best performing model run
    model = load_model('model_best.hdf5')
    
    # Evaulate the model
    eval_model=model.evaluate(featureTest, targetTest)
    eval_model
    
    # Plot the NN accuracy over each epoch
    fig, ax = plt.subplots()
    ax.plot(history.history['loss'])
    ax.plot(history.history['val_loss'])
    ax.legend(['Training','Validation'])
    plt.show()
    # fig.savefig('NN_modelTrainingHistory.png')
    
    # Generate predictions for the test period
    modelPrediction = model.predict(featureTest, batch_size=32)

    # And now use the threhsold to decide y or n
    modelPredThresh=1*(modelPrediction >= classThreshold)

    # Compute confusion matrix
    cnfMatrix = confusion_matrix(targetTest, modelPredThresh)
    np.set_printoptions(precision=2)

    # Plot normalized confusion matrix
    classNames = ['bad data point','good data point']
    plotConfusionMatrix(cnfMatrix, classes=classNames,normalize=True,
                      title='Confusion matrix, with normalization')
    plt.show()
    
    # Generate model predictions for the validation period
    modelPrediction = model.predict(predictFeatures, batch_size=32)
    # Generate the y/n prediction given the above model threshold value (right now = 0.9)
    modelPredThresh=1*(modelPrediction >= classThreshold)
    
    modelOut = pd.DataFrame()
    modelOut['primary'] = predictFeatures['PRIMARY']
    modelOut['modelPrediction']=modelPrediction
    modelOut['InvModelPrediction']=1-modelPrediction
    modelOut['goodPtsPrediction']=modelPredThresh
    modelOut['target'] = predictTargets['TARGET']
    modelOut['station'] = predictStations['STATION_ID']
    
    return modelOut, model

# ________________________________________________________________________________________________________________


# Create the station list 
# Grab the station list by default from the directory of files
dirList = os.listdir( '/Users/Samantha.Longridge/Documents/Python Scripts/WLAI/GitHub Files/data/train')
stationList = []
for file in dirList:
    if not file.startswith('.'):
        stationList.append(file[:7])
        
# removing south beach for now - has some sort of issue, was an issue, has been corrected and is commented out because it has been fixed
# stationList.remove('9435380')

# Create performance dataframe
# Make the modelPerformanceDataFrame to store info about each station and error stats
modelPerformance = pd.DataFrame()
stationList.insert(0,'allStations')
modelPerformance['stationList'] = stationList
modelPerformance.set_index('stationList',inplace = True)

modelPerformance['trainingPts'] = np.nan
modelPerformance['trainingFractionBad'] = np.nan
modelPerformance['valPts'] = np.nan
modelPerformance['valFractionBad'] = np.nan

modelPerformance['accuracy'] = np.nan
modelPerformance['badPtAccuracy'] = np.nan
modelPerformance['BSS'] = np.nan
modelPerformance['AROC'] = np.nan

# ________________________________________________________________________________________________________________

# for each station in my list of stations I want to include, load the training data and validation data.  
# Also - resample to create data sets that are 10% bad data points and then cat together

# For each station in list load the training and validation data, print the data counts, resample and cat together
trainAll = pd.DataFrame()
validateAll = pd.DataFrame()
allTrainData=0
allTrainBadData=0
allValData=0
allValBadData=0

for station in stationList[1:]:
    print(station)
    # load the training data
    trainIn = loadCleanedData(station, 'train', '/Users/Samantha.Longridge/Documents/Python Scripts/WLAI/GitHub Files/data')
    validateIn = loadCleanedData(station, 'validation', '/Users/Samantha.Longridge/Documents/Python Scripts/WLAI/GitHub Files/data')
    
    # Remove all cases where PRIMARY_TRUE = 0, since then the primary data is missing, and we aren't assessing its quality
    trainIn.drop(trainIn[trainIn['PRIMARY_TRUE'] == 0].index, inplace = True) 
    validateIn.drop(validateIn[validateIn['PRIMARY_TRUE'] == 0].index, inplace = True) 
    
    # primary_true = 1 means that data was missing, doesn't evaluate missing data, because missing data is obviously bad
    
    # resample the training data
    trainResample = resampleGoodPointsSetNum(200000, trainIn) # reducing the number of good data points to bad data points  
    
    # 90% good : 10% bad 
    
    #assess the data counts
    [totalTrainData,badTrainData] = assessTrainTestData(trainResample)
    [totalValData,badValData] = assessTrainTestData(validateIn)
    
    #load into the dataframe
    modelPerformance.at[station,'trainingPts'] = totalTrainData
    modelPerformance.at[station,'trainingFractionBad'] = badTrainData/totalTrainData
    modelPerformance.at[station,'valPts'] = totalValData
    modelPerformance.at[station,'valFractionBad'] = badValData/totalValData
    
    #Cat into a single dataframe
    trainAll = pd.concat([trainAll, trainResample])
    validateAll = pd.concat([validateAll, validateIn])
    
    #Calculate the total data numbers
    allTrainData= allTrainData + totalTrainData
    allTrainBadData= allTrainBadData+badTrainData
    allValData= allValData + totalValData
    allValBadData= allValBadData + badValData


#Add the total data numbers to the dataframe
modelPerformance.at['allStations','trainingPts'] = allTrainData
modelPerformance.at['allStations','trainingFractionBad'] = allTrainBadData/allTrainData
modelPerformance.at['allStations','valPts'] = allValData
modelPerformance.at['allStations','valFractionBad'] = allValBadData/allValData

# Shuffle data                 # shuffles the data so that it's not in order? / maybe consider note shuffling the data?
trainRand=shuffle(trainAll)
validateRand=shuffle(validateAll)


modelPerformance

# ________________________________________________________________________________________________________________

# Now set up the model inputs 

featureNames = ['PRIMARY','PRIMARY_SIGMA','PRIMARY_SIGMA_TRUE','PRIMARY_RESIDUAL','BACKUP','BACKUP_TRUE','PREDICTION',]

featureTrain=trainRand.loc[:, featureNames]
featureVal=validateRand.loc[:, featureNames]

targetTrain=trainRand.loc[:,['TARGET']]
targetVal=validateRand.loc[:,['TARGET']]


predictFeatures = validateAll.loc[:, featureNames]
predictTargets = validateAll.loc[:,['TARGET']]
predictStations = validateAll.loc[:,['STATION_ID']]

# ________________________________________________________________________________________________________________

#NN approach - 10% no - Simple model - with 32x2 and 0.25% dropout


epochCount = 30
batchCount = 256
classThreshold = .50
numFeatures = len(featureNames)
modelOut_simple, model_simple = runNNmodel(featureTrain, targetTrain, featureVal, targetVal, predictFeatures, predictTargets,
               predictStations, epochCount, batchCount, classThreshold, numFeatures)

# ________________________________________________________________________________________________________________


#The climatology comparison for the BSS will be accross all stations no matter what
targetClimIn = targetTrain.to_numpy()

#Loop through first all stations and then by station to calculate error stats
for stationNum in modelPerformance.index:
      
    if stationNum == 'allStations':  
        #define variables
        goodPtsPredIn = modelOut_simple['goodPtsPrediction']
        obsTargetIn = modelOut_simple['target'].to_numpy()
        onlyBadPoints = modelOut_simple[modelOut_simple['target'] == 0]
        goodPtsPredOnlyBadIn = onlyBadPoints['goodPtsPrediction']
        obsTargetOnlyBadIn = onlyBadPoints['target']
        modPredIn = modelOut_simple['modelPrediction'].to_numpy()   
    else:
        #define station specific variables
        goodPtsPredIn = modelOut_simple[modelOut_simple['station']==int(stationNum)]['goodPtsPrediction']
        obsTargetIn = modelOut_simple[modelOut_simple['station']==int(stationNum)]['target'].to_numpy()
        onlyBadPoints = modelOut_simple[modelOut_simple['target'] == 0]
        goodPtsPredOnlyBadIn = onlyBadPoints[onlyBadPoints['station']==int(stationNum)]['goodPtsPrediction']
        obsTargetOnlyBadIn = onlyBadPoints[onlyBadPoints['station']==int(stationNum)]['target']
        modPredIn =modelOut_simple[modelOut_simple['station']==int(stationNum)]['modelPrediction'].to_numpy()   
        
    
    #Calculate the values
    modelPerformance.at[stationNum,'accuracy'] = 1 - np.sum(np.abs(goodPtsPredIn - obsTargetIn))/len(goodPtsPredIn) 

    #Bad Pt Accuracy
    modelPerformance.at[stationNum,'badPtAccuracy'] = 1 - np.sum(np.abs(goodPtsPredOnlyBadIn - obsTargetOnlyBadIn  ))/len(obsTargetOnlyBadIn) 

    #BSS
    modelPerformance.at[stationNum,'BSS'] = BSS(targetClimIn, modPredIn, obsTargetIn)

    #AROC
    modelPerformance.at[stationNum,'AROC'] = roc(obsTargetIn,np.reshape(modPredIn,-1))

        
modelPerformance    

