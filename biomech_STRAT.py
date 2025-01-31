# -*- coding: utf-8 -*-
"""
This script finds the classification results for a biomechanically structured
recognition scheme. All activities are first divided into "static" and "dynamic".
The "static" activities are then divided into "vigilance" and "resting". The
"dynamic" ones are then divided into "foraging" and "running".
At each of these three classification steps, the best classifier is chosen out
of three possibilities: (1) Naïve-Bayes, (2) LDA, (3) Logistic Regression, 
(4) SVM (linear), and (5) MLP1_10. The results will be stored for all the 
5x5x5 = 125 combinations of models for each fold and for each individual.

The cross-validation method used is stratified k-fold cross-validation.

Written:    03 Sep, 2017
            Pritish Chakravarty
            LMAM, EPFL
            
Edited:     15 Sep, 2017
            (added part towards the end that will write the results for each
            model (aggregated as well as mean+/- std) to two separate CSV files
            for each feature set for which the biomechanical models were built)
"""

#%% Importing required libraries

import pickle
import pandas
import time

# importing the models to be tested
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# importing functions I wrote
from crossValidationFunctions import kFoldResults_biomechanicalModel

# importing some tools from numpy
from numpy import array
from numpy import ones


#%% Prepare the models to be tested

models = []
models.append(('NB', GaussianNB()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('LR', LogisticRegression()))
models.append(('SVM', SVC(kernel='linear'))) # uses a linear kernel


#%% Loading Features C2 (total 5 features) and preparing "static" and "dynamic" labels

featuresC2_names = ['meanX','stdNorm','fftPeakPowerAvg','class','meerkatNum']

# load the feature matrix
dataframe = pandas.read_csv('featuresC2_biomechanical.csv',names = featuresC2_names)
Array = dataframe.values
X = Array[:,:-2] # feature matrix (excluding true labels and meerkat number)
Y = Array[:,-2] # true labels in numerical format (i.e. 1's, 2's, 3's and 4's)
N = Array[:,-1] # meerkat number (1-8 for the time being)

# creating the static-dynamic labels
Y_SD = ones(len(Y)) # an array of numeric static-dynamic labels derived from "Y". Contains "1" for static and "2" for dynamic
Y_SD[[idx for idx,y in enumerate(Y) if y==3 or y==4]] = 2 # wherever we detect a "3" (foraging) or a "4" (running), change the "1" to a "2" in "Y_SD"


#%% Create different combinations of features to find which one gives the best performance, and which is the most minimalist (and still gives good results)

featureIdx = [] # this will be a list of all feature-combinations we would like to try, and each entry will be a list of three arrays which specify the indices of the features to be used for the three stages of classification. The first array corresponds to the indices of the features to be used for static VS dynamic classification, the second to vigilance VS resting, and the third to foraging VS running
featureIdx.append([array([0,1]), 0, array([1,2])]) # [STATIC_DYNAMIC: meanX,stdNorm]; [VIG_REST: meanX]; [FRG_RUN: stdNorm,fftPeakPowerAvg]
featureIdx.append([array([0,1]), array([0,1]), array([1,2])]) # [STATIC_DYNAMIC: meanX,stdNorm]; [VIG_REST: meanX,stdNorm]; [FRG_RUN: stdNorm,fftPeakPowerAvg]


#%% Compute classification results with 5x5x5 models (each of the 5 model at each of the 3 branches of the biomechanically constructed classification tree)

results = [] # a list of all results. It has the same number of rows as those in "featureIdx". Each element of this list is again a list, containing the results of all the 5x5x5 models along with the names of the models used in each iteration

count = 0 # the feature-set count for which all models were successfully run

for F in featureIdx:         # for each set of features

    start_time0 = time.time() # start time for the computations to be done for this feature set

    results_temp = [] # will store all the 5x5x5 results for the current set of features. Each element of this list contains: ['M1-M2-M3', aggregatedResults, meanResults, foldWiseResults], where 'M1_M2_M3' can be 'NB_LR_SVM', for example
    
    for i in range(len(models)):        # for each static VS dynamic model
        for j in range(len(models)):            # for each vig VS resting model
            for k in range(len(models)):                # for each foraging VS running model
            
                threeModels = [models[i], models[j], models[k]]
                
                # finding the results
                aggregated, meanFold, foldWise = kFoldResults_biomechanicalModel(threeModels,F,X,Y,Y_SD,numFolds=10,balanceMethod='none')
                
                # building a string containing the name of the three models used in this iteration
                separator = '-'
                s = (models[i][0], models[j][0], models[k][0])
                S = separator.join(s)
                
                # finally, appending the model information and the results for this iteration to "results_temp"
                results_temp.append([S, aggregated, meanFold, foldWise])
                
                msg = "%i of %i models run for this feature set" % (i*len(models)**2 + j*len(models) + k+1, len(models)**3)
                print(msg)
                
        # end of model-combination-wise computation (total 5x5x5 model-combinations were tested) for the current feature-set
    
    results.append(results_temp) # appending the results for all the 5x5x5 models for this set of features to the overall list named "results"
    
    count += 1 # updating the feature-set count for which all models were successfully run
    
    # finally, printing a status update saying that all models for the current feature set were successfully run
    msg = "%i of %i feature sets analysed" % (count, len(featureIdx))
    print(msg)
    
    msg = "Time taken for this feature set: %f s" % (time.time() - start_time0)
    print(msg)
    
    
    
    # end of model-combination wise computation for all feature sets
    

#%% Saving the results

with open('results_biomechModels_kFold.pickle', 'wb') as f:
    pickle.dump([results, featureIdx, featuresC2_names], f)


#==============================================================================
# # Loading the saved results
# with open('results_biomechModels_kFold.pickle','rb') as f:
#     results,featureIdx,featuresC2_names = pickle.load(f)
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
    for name,aggResults,meanResults,FoldResults in modelResults:
        
        modelNames.append(name)
        
        for j in range(numClasses):
            resultsMatrix[count,j*2+0] = aggResults.sensitivity[j]
            resultsMatrix[count,j*2+1] = aggResults.precision[j]
            # end of class-wise computation
            
        resultsMatrix[count,-1] = aggResults.accuracy # overall accuracy for current model
            
        count += 1 # go to next unfilled row of "resultsMatrix" before accessing the results of the next model
                
        # end of model-wise computation
        
    ff = featureSetIdx + 1
    filename = 'table[03a]_agg_FS' + ('%ia.csv' % ff)
    
    savetxt(filename, resultsMatrix, delimiter=',', fmt='%.1f')
    
    # end of feature-set-wise computation


#%% Writing mean and standard deviation of performance metrics (across all folds) to a CSV file

import csv

for featureSetIdx in range(len(featureIdx)):
    
    ff = featureSetIdx + 1
    filename = 'table[03a]_meanstd_FS' + ('%ib_meanstd.csv' % ff)
    
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