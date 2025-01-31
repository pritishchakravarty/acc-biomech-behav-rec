# -*- coding: utf-8 -*-
"""
This script creates a number "n_subsets" of randomly sampled subsets of the
entire data contained in "X" (the feature matrix) and "Y" (the list of labels)
such that all classes in each subset are balanced. The "class-balancing method"
used is "EasyEnsemble" in the "ensemble of samplers" category in the package 
"imbalanced-learn" (version 0.2.1).
Once these subsets are created, the desired sequence of three models is run on 
each subset (i.e. kFold cross-validation is performed on each of these subsets). 
The aggregated confusion matrices (for the 10 test folds created for each subset)
from each subset will then be aggregated across all subsets to give a "twice-
aggregated" confusion matrix from which the usual performance metrics (sensitivity, 
specificity, and precision) will be calculated and reported.

These results will help answer the question: simplifying for imbalances in 
class-distribution and differences between individual meerkats (in carrying out
the same activities), how separable are the four different activities given 
the set of features used? And how simple a model can distinguish between them?

Written:    08 Sep, 2017
            Pritish Chakravarty
            LMAM, EPFL
            
Edited:     15 Sep, 2017
            (added part towards the end that will write the results for each
            model (aggregated as well as mean +/- std) to two separate CSV files
            for each feature set for which the biomechanical models were built)
            
Edited:     17 Sep, 2017
            (added part that finds the mean and std of performance metrics
            across all folds and all subsets)

"""

#%% Importing the required libraries and functions

import pandas
import pickle
import time

from imblearn.ensemble import EasyEnsemble

# importing the models to be tested
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from crossValidationFunctions import kFoldResults_biomechanicalModel
from crossValidationFunctions import computePerformanceMetrics
from crossValidationFunctions import initialiseStruct

from numpy import ones
from numpy import zeros
from numpy import unique
from numpy import array
from numpy import asarray
from numpy import mean
from numpy import std


#%% Loading the dataset: features C2 (total 5 features); preparing "static" and "dynamic" labels

featuresC2_names = ['meanX','stdNorm','fftPeakPowerAvg','class','meerkatNum']

# load the feature matrix
dataframe = pandas.read_csv('featuresC2_biomechanical.csv',names = featuresC2_names)
Array = dataframe.values
X = Array[:,:-2] # feature matrix (excluding true labels and meerkat number)
Y = Array[:,-2] # true labels in numerical format (i.e. 1's, 2's, 3's and 4's)
N = Array[:,-1] # meerkat number (1-8 for the time being)

numClasses = len(unique(Y))


#%% Loading models that can be candidates for each of the three stages of classification (StaticDynamic-VigRest-FrgRun)

models = []
models.append(('NB', GaussianNB()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('LR', LogisticRegression()))
models.append(('SVM', SVC(kernel='linear'))) # uses a linear kernel

# # list of all feature sets
#==============================================================================
featureIdx = [] # this will be a list of all feature-combinations we would like to try, and each entry will be a list of three arrays which specify the indices of the features to be used for the three stages of classification. The first array corresponds to the indices of the features to be used for static VS dynamic classification, the second to vigilance VS resting, and the third to foraging VS running
featureIdx.append([array([0,1]), 0, array([1,2])]) # [STATIC_DYNAMIC: meanX,stdNorm]; [VIG_REST: meanX]; [FRG_RUN: stdNorm,fftPeakPowerAvg]
featureIdx.append([array([0,1]), array([0,1]), array([1,2])]) # [STATIC_DYNAMIC: meanX,stdNorm]; [VIG_REST: meanX,stdNorm]; [FRG_RUN: stdNorm,fftPeakPowerAvg]


#%% Generating an ensemble of "n_subsets" randomly undersampled subsets of "X" with balanced classes

ee = EasyEnsemble(random_state=0, n_subsets=10)
x_resampled, y_resampled = ee.fit_sample(X,Y)
# the shape of x_resampled is: [n_subsets, no. of data points per subset, no. of features for each data point]

#==============================================================================
# print(sorted(Counter(y_resampled[0]).items())) # one observes that the number of instances of each class is equal to the number of instances of the least-frequent class (which is "Running" in our case)
#==============================================================================


#%% Running a tri-model combination (StaticDynamic-VigRest-FrgRun) on each subset and accumulating the confusion matrix (for a given set of features for each stage of classification)

results = [] # a list of all results. It has the same number of rows as those in "featureIdx". Each element of this list is again a list, containing the results of all the 5x5x5 models along with the names of the models used in each iteration

for featureSetIdx in range(len(featureIdx)):
    
    start_time0 = time.time() # start time for the computations to be done for this feature set
    
    F = featureIdx[featureSetIdx] # the current feature set for which results will be computed in this iteration

    results_temp = [] # will store all the 5x5x5 results for the chosen set of features. Each element of this list contains: ['M1-M2-M3', confusionMatrix, sensitivity, specificity, precision], where 'M1-M2-M3' can be 'NB_LR_SVM', for example
    
    for i in range(len(models)):        # for each static VS dynamic model
            for j in range(len(models)):            # for each vig VS resting model
                for k in range(len(models)):                # for each foraging VS running model
                
                    threeModels = [models[i], models[j], models[k]]
                    
                    # various initialisations
                    aggSubset = initialiseStruct() # will store the results from the confusion matrices aggregated across subsets and folds within subsets
                    meanSubset = initialiseStruct() # will store the mean and std results from the confusion matrix from each fold within each subset
                    allSensitivities = [] # will store the sensitivity calculated at each fold of each subset
                    allSpecificities = [] # will store the specificity calculated at each fold of each subset
                    allPrecisions = []  # will store the precision calculated at each fold of each subset
                    allAccuracies = []  # will store the accuracy calculated at each fold of each subset
                    
                    C = zeros([numClasses,numClasses]) # initialising the overall "twice-aggregated" confusion matrix (first aggregated over each fold of each class-balanced subset, and then aggregated over each such subset)
    
                    for sbstNum in range(x_resampled.shape[0]):   # for each class-balanced subset of "X"
                    
                        x = x_resampled[sbstNum,:,:] # the feature matrix for this iteration
                        y = y_resampled[sbstNum,:] # the vector of labels corresponding to "x"
                        
                        # creating the static-dynamic labels for the data points being used in this iteration
                        y_SD = ones(len(y)) # an array of numeric static-dynamic labels derived from "y". Contains "1" for static and "2" for dynamic
                        y_SD[[idx for idx,actvtLabel in enumerate(y) if actvtLabel==3 or actvtLabel==4]] = 2 # wherever we detect a "3" (foraging) or a "4" (running), change the "1" to a "2" in "y_SD"
                        
                        # finding the results of classification for this subset
                        aggregated, meanFold, foldWise = kFoldResults_biomechanicalModel(threeModels,F,x,y,y_SD,numFolds=10)
                        
                        # saving the performance metrics obtained from each fold in this subset
                        allSensitivities.append(foldWise.sensitivity)
                        allSpecificities.append(foldWise.specificity)
                        allPrecisions.append(foldWise.precision)
                        allAccuracies.append(foldWise.accuracy)
                        
                        # updating the overall confusion matrix (across all subsets)
                        C += aggregated.confMat
                        
                        # END of computation of overall confusion matrix (across all class-balanced subsets) for current model
                        
                    # now computing performance-wise metrics for the final confusion matrix produced 
                    sensitivity,specificity,precision,accuracy = computePerformanceMetrics(C)
                    
                    #-----------SAVING THE RESULTS IN VARIABLES----------------
                    
                    # saving the aggregated results across all subsets for the current tri-model-combination
                    aggSubset.confMat = C
                    aggSubset.sensitivity = sensitivity
                    aggSubset.specificity = specificity
                    aggSubset.precision = precision
                    aggSubset.accuracy = accuracy
                    
                    
                    # computing the means and standard deviations of the three performance metrics across all folds of all subsets
                    allSensitivities = asarray(allSensitivities)
                    allSpecificities = asarray(allSpecificities)
                    allPrecisions = asarray(allPrecisions)
                    allAccuracies = asarray(allAccuracies)
                    
                    sensitivity_mean = zeros([numClasses])
                    sensitivity_std = zeros([numClasses])
                    specificity_mean = zeros([numClasses])
                    specificity_std = zeros([numClasses])
                    precision_mean = zeros([numClasses])
                    precision_std = zeros([numClasses])
                    accuracy_mean = zeros([numClasses])
                    accuracy_std = zeros([numClasses])
                    
                    for actvtNum in range(numClasses):
                        
                        # calculating the means of the performance metrics across all folds and all subsets
                        sensitivity_mean[actvtNum] = mean(allSensitivities[:,:,actvtNum])
                        specificity_mean[actvtNum] = mean(allSpecificities[:,:,actvtNum])
                        precision_mean[actvtNum] = mean(allPrecisions[:,:,actvtNum])
                        
                        
                        # calculating the standard deviations of the performance metrics across all folds and all subsets
                        sensitivity_std[actvtNum] = std(allSensitivities[:,:,actvtNum])
                        specificity_std[actvtNum] = std(allSpecificities[:,:,actvtNum])
                        precision_std[actvtNum] = std(allPrecisions[:,:,actvtNum])
                        
                        # END of class--wise performance-metric calculation
                    
                    accuracy_mean = mean(allAccuracies)
                    accuracy_std = std(allAccuracies)
                    
                    meanSubset.sensitivity_mean = sensitivity_mean
                    meanSubset.sensitivity_std = sensitivity_std
                    meanSubset.specificity_mean = specificity_mean
                    meanSubset.specificity_std = specificity_std
                    meanSubset.precision_mean = precision_mean
                    meanSubset.precision_std = precision_std
                    meanSubset.accuracy_mean = accuracy_mean
                    meanSubset.accuracy_std = accuracy_std
                    
                    #----------------------------------------------------------
                                        
                    # building a string containing the name of the three models used in this iteration
                    separator = '-'
                    s = (models[i][0], models[j][0], models[k][0])
                    S = separator.join(s)
                    
                    # finally, appending the model information and the results for this iteration to "results"
                    results_temp.append([S, aggSubset, meanSubset]) 
                    """CAUTION! Make sure "aggSubset" and "meanSubset" are 
                    initialised within this loop. If they are initialised 
                    outside this loop, rewriting their values in each iteration
                    of this loop actually rewrites the values that were written
                    in "results_temp" and also in "results"! Thus, the first 
                    time I ran this code, ALL results for ALL 125 models for 
                    each feature-set were identical, even the standard 
                    deviations! This is because each result stored in 
                    "results_temp" and "results" simply got updated to the 
                    latest values in "aggSubset" and "meanSubset" each time we 
                    tried to "replace" them in each iteration. Funny Python 
                    behaviour."""
                    
                    msg = "%i of %i models run for this feature set" % (i*len(models)**2 + j*len(models) + k+1, len(models)**3)
                    print(msg)
                 


    results.append(results_temp) # appending the results for all the 5x5x5 models for this set of features to the overall list named "results"
    
    count = featureSetIdx + 1
    # finally, printing a status update saying that all models for the current feature set were successfully run
    msg = "%i of %i feature sets analysed" % (count, len(featureIdx))
    print(msg)
    
    msg = "Time taken for this feature set: %f s" % (time.time() - start_time0)
    print(msg)


#%% Saving the results

with open('results_balanced_undersampleBEFORE_kFold_biomechModels.pickle', 'wb') as f:
    pickle.dump([results, featureIdx, featuresC2_names], f)
    
#==============================================================================
# # Loading the results
# with open('results_balanced_undersampleBEFORE_kFold_biomechModels.pickle', 'rb') as f:
#     results, featureIdx, featuresC2_names = pickle.load(f)
#==============================================================================
    

#%% Writing aggregated results for all models to a CSV file

from numpy import zeros
from numpy import savetxt

numClasses = 4 # the four classes are: vigilance, resting, foraging, running

for featureSetIdx in range(len(featureIdx)):
    
    modelResults = results[featureSetIdx] # contains the results for this feature-set
    
    resultsMatrix = zeros([len(modelResults), numClasses*2+1]) # will contain the results for each of the 5*5*5 models in matrix format. Each model corresponds to one row, and each row has a total of 9 columns: 2 columns (sensitivity, precision) for each class, and one column for the overall accuracy of the model
    
    modelNames = [] # will store the names of all the models
    
    count = 0 # counts the number of rows of "resultsMatrix" that have already been filled
    for name,AggSubsetResults,meanSubsetResults in modelResults:
    
        for j in range(numClasses):
            resultsMatrix[count,j*2+0] = AggSubsetResults.sensitivity[j]
            resultsMatrix[count,j*2+1] = AggSubsetResults.precision[j]
            
        resultsMatrix[count,-1] = AggSubsetResults.accuracy # overall accuracy across all subsets for current model
        
        count += 1 # go to next unfilled row of "resultsMatrix" before entering the next iteration of the first "for" loop # go to next unfilled row of "resultsMatrix" before accessing the results of the next model
        
        # END of model-wise computation
        
    ff = featureSetIdx + 1
    filename = 'results[04a1]_aggregated_FS' + ('%ia.csv' % ff)
    
    savetxt(filename, resultsMatrix, delimiter=',', fmt='%.1f')
    
    # END of feature-set-wise computation
    
    
#%% Writing mean and standard deviation of performance metrics (across all folds and all subsets) to a CSV file

import csv

for featureSetIdx in range(len(featureIdx)):
    
    ff = featureSetIdx + 1
    filename = 'table[04a1]_FS' + ('%ib_meanstd.csv' % ff)
    
    modelResults = results[featureSetIdx] # contains the results for this feature-set
    
    resultsMatrix = zeros([len(modelResults),numClasses*2 + 1]) # will contain the results in matrix format at the end
    # one row for each model for which results were obtained with this feature set, and 2 columns (sensitivity (mean ± std), precision (mean ± std)) for each class, with one columns at the end (for overall accuracy (mean ± std))
    
    for I in range(len(modelResults)):
        
        Ithmodelstring = [] # string containing the class-wise results for this model
        
        for j in range(numClasses):
            
            Sm = modelResults[I][2].sensitivity_mean[j] # mean sensitivity
            Sstd = modelResults[I][2].sensitivity_std[j] # std of sensitivity
            Pm = modelResults[I][2].precision_mean[j] # mean precision
            Pstd = modelResults[I][2].precision_std[j] # std of precision
            
            Ithmodelstring.append(('%.1f' % Sm)+' ± '+('%.1f' % Sstd))
            Ithmodelstring.append(('%.1f' % Pm)+' ± '+('%.1f' % Pstd))
            
            # end of class-wise computation
            
        # adding the overall model accuracy (mean ± std) at the end for each feature-set
        Am = modelResults[I][2].accuracy_mean
        Astd = modelResults[I][2].accuracy_std
        Ithmodelstring.append(('%.1f' % Am)+' ± '+('%.1f' % Astd))
        
        # writing this row of results for the current model to a CSV file
        with open(filename,'a',newline='') as resultFile:
            wr = csv.writer(resultFile, dialect='excel')
            wr.writerow(Ithmodelstring)
        
        # end of model-wise computation
    
    # end of feature-set-wise computation
            