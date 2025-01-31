#%% Importing the required libraries and functions

import pandas
import pickle

# importing the models to be tested
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier # see http://scikit-learn.org/stable/modules/neural_networks_supervised.html
from sklearn.svm import SVC

from numpy import savetxt
from numpy import asarray
from numpy import zeros

from crossValidationFunctions import prebalancedKfoldResults


#%% Prepare the models to be tested

models = []
models.append(('NaiveBayes', GaussianNB()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('QDA', QuadraticDiscriminantAnalysis()))
models.append(('KNN5', KNeighborsClassifier(n_neighbors=5))) # no. of neighbors = 5
models.append(('LogReg', LogisticRegression()))
models.append(('DecTree', DecisionTreeClassifier()))
models.append(('RndForest10', RandomForestClassifier(n_estimators=10))) # no. of trees in the forest = 10
models.append(('MLP1_10', MLPClassifier(hidden_layer_sizes=(10,)))) # 1 hidden layer with 10 hidden units
models.append(('SVM_linear', SVC(kernel='linear'))) # uses a linear kernel
models.append(('SVM_rbf', SVC(kernel='rbf'))) # uses a radial basis function (Gaussian) kernel


#%% FeaturesA_statistical: Loading data, running models

featuresA_names = ['meanX', 'meanNorm', 'stdX', 'stdNorm', 'skewnessX', 'skewnessNorm', 'kurtosisX', 'kurtosisNorm', 'maxX', 'maxNorm', 'minX', 'minNorm', 'autocorrX', 'autocorrNorm', 'trendX', 'trendNorm', 'class','meerkatNum'] # feature names, also used as headers for columns in feature matrix

# load the feature matrix
dataframe = pandas.read_csv('featuresA_statistical.csv',names = featuresA_names)
Array = dataframe.values
X = Array[:,:-2] # feature matrix (excluding true labels and meerkat number)
Y = Array[:,-2] # true labels in numerical format (i.e. 1's, 2's, 3's and 4's)
N = Array[:,-1] # meerkat number (1-8 for the time being)

#------------------------------------------------------------------------------

# KEEPING ONLY THE THREE BEST FEATURES (found using the five feature selection methods in "topKfeatures")
top3_A = ['stdX', 'stdNorm', 'minNorm']
# finding the indices of the features in "top3_A" in "allFeaturesA_names"
I3 = asarray([idx for i in range(len(top3_A)) for idx in range(len(featuresA_names)) if top3_A[i]==featuresA_names[idx]])
# updating "X" so that it contains only the top 3 features and deletes everything else
X = X[:,I3]

#------------------------------------------------------------------------------

# evaluate each model in turn, and store the results of each
resultsA = prebalancedKfoldResults(X,Y,models)
    
print("The results for the statistical features (features A) have been computed")
    
    
#%% FeaturesB_humanActivity: Loading data, running models

featuresB_names = ['meanX','meanNorm','stdX','stdNorm','stdPC1','meanRectHPX','meanRectHPNorm','meanRectHPPC1','prinFreqX','prinFreqNorm','prinFreqPC1','fftEntropyX','fftEntropyNorm','fftEntropyPC1','fftMag1X','fftMag2X','fftMag3X','fftMag1Norm','fftMag2Norm','fftMag3Norm','fftMag1PC1','fftMag2PC1','fftMag3PC1','class','meerkatNum']

# load the feature matrix
dataframe = pandas.read_csv('featuresB_humanActivity.csv',names = featuresB_names)
Array = dataframe.values
X = Array[:,:-2] # feature matrix (excluding true labels and meerkat number)
Y = Array[:,-2] # true labels in numerical format (i.e. 1's, 2's, 3's and 4's)
N = Array[:,-1] # meerkat number (1-8 for the time being)

#------------------------------------------------------------------------------

# KEEPING ONLY THE THREE BEST FEATURES (found using the five feature selection methods in "topKfeatures")
top3_B = ['stdX', 'stdNorm', 'meanRectHPNorm'] # going from 8 meerkats to 11 meerkats, "meanRectHPNorm" replaced "meanX". "stdX" and "stdNorm" remain unchanged
# finding the indices of the features in "top3_B" in "allFeaturesB_names"
I3 = asarray([idx for i in range(len(top3_B)) for idx in range(len(featuresB_names)) if top3_B[i]==featuresB_names[idx]])
# updating "X" so that it contains only the top 3 features and deletes everything else
X = X[:,I3]

#------------------------------------------------------------------------------

# evaluate each model in turn, and store the results of each
resultsB = prebalancedKfoldResults(X,Y,models)
    
print("The results for the human activity features (features B) have been computed")


#%% FeaturesABtogether: Loading data, running models

# loading Features ABtogether
allFeaturesABtogether_names = ['meanX', 'meanNorm', 'stdX', 'stdNorm', 'stdPC1', 'skewnessX', 'skewnessNorm', 'kurtosisX', 'kurtosisNorm', 'maxX', 'maxNorm', 'minX', 'minNorm', 'autocorrX', 'autocorrNorm', 'trendX', 'trendNorm', 'meanRectHPX','meanRectHPNorm','meanRectHPPC1','prinFreqX','prinFreqNorm','prinFreqPC1','fftEntropyX','fftEntropyNorm','fftEntropyPC1','fftMag1X','fftMag2X','fftMag3X','fftMag1Norm','fftMag2Norm','fftMag3Norm','fftMag1PC1','fftMag2PC1','fftMag3PC1','class','meerkatNum'] # feature names, also used as headers for columns in feature matrix

# load the feature matrix
dataframe = pandas.read_csv('featuresABtogether.csv',names = allFeaturesABtogether_names)
array = dataframe.values
X = array[:,:-2] # feature matrix (excluding true labels and meerkat number)
Y = array[:,-2] # true labels in numerical format (i.e. 1's, 2's, 3's and 4's)
N = array[:,-1] # meerkat number (1-8 for the time being)

#------------------------------------------------------------------------------

# KEEPING ONLY THE THREE BEST FEATURES (found using the five feature selection methods in "topKfeatures")
top3_ABtogether = ['stdX', 'minX', 'meanRectHPNorm'] # going from feature-set B to ABtogether, "stdNorm" gets replaced by "minX". "stdX" and "meanRectHPNorm" remain unchanged
# finding the indices of the features in "top3_B" in "allFeaturesB_names"
I3 = asarray([idx for i in range(len(top3_ABtogether)) for idx in range(len(allFeaturesABtogether_names)) if top3_ABtogether[i]==allFeaturesABtogether_names[idx]])
# updating "X" so that it contains only the top 3 features and deletes everything else
X = X[:,I3]

#------------------------------------------------------------------------------

resultsABtogether = prebalancedKfoldResults(X,Y,models)
    
print("The results for the top 3 overall features (features ABtogether) have been computed")


#%% FeaturesC2_biomechanical: Loading data, running models

featuresC_names = ['meanX','stdNorm','fftPeakPowerAvg','class','meerkatNum']

# load the feature matrix
dataframe = pandas.read_csv('featuresC2_biomechanical.csv',names = featuresC_names)
Array = dataframe.values
X = Array[:,:-2] # feature matrix (excluding true labels and meerkat number)
Y = Array[:,-2] # true labels in numerical format (i.e. 1's, 2's, 3's and 4's)
N = Array[:,-1] # meerkat number (1-8 for the time being)

#------------------------------------------------------------------------------

# KEEPING ONLY THE THREE BEST FEATURES (found using the five feature selection methods in "topKfeatures")
top3_C = ['meanX', 'stdNorm', 'fftPeakPowerAvg']
# finding the indices of the features in "top3_A" in "allFeaturesA_names"
I3 = asarray([idx for i in range(len(top3_C)) for idx in range(len(featuresC_names)) if top3_C[i]==featuresC_names[idx]])
# updating "X" so that it contains only the top 3 features and deletes everything else
X = X[:,I3]

#------------------------------------------------------------------------------

resultsC = prebalancedKfoldResults(X,Y,models)
    
print("The fold-wise results for the biomechanical features (features C2) have been computed")


#%% Save all results

with open('results_balanced_undersampleBEFORE_kFold_MLmodels_top3Features.pickle', 'wb') as f:
    pickle.dump([resultsA, resultsB, resultsABtogether, resultsC], f)
    
#==============================================================================
# # Loading the file
# with open('results_balanced_undersampleBEFORE_kFold_MLmodels_top3Features.pickle','rb') as f:
#     resultsA, resultsB, resultsABtogether, resultsC = pickle.load(f)
#==============================================================================


#%% Writing aggregated results to a CSV file

numClasses = 4
numFeatureSets = 4

resultsMatrix = [] # will contain the results in matrix format at the end

for I in range(len(models)):
    
    thisModelMatrix = zeros([numFeatureSets,numClasses*2+1]) # one row for each feature set tested with this model, and 2 columns (sensitivity, precision) for each class, with one column at the end (for overall accuracy)
        
    for j in range(numClasses):
        # row number = 0 in "thisModelMatrix" signifies that the results are for featuresA (statistical)
        thisModelMatrix[0,j*2+0] = resultsA[I][1].sensitivity[j]
        thisModelMatrix[0,j*2+1] = resultsA[I][1].precision[j]
        
        # row number = 1 in "thisModelMatrix" signifies that the results are for featuresB (humanActivity)
        thisModelMatrix[1,j*2+0] = resultsB[I][1].sensitivity[j]
        thisModelMatrix[1,j*2+1] = resultsB[I][1].precision[j]
        
        # row number = 2 in "thisModelMatrix" signifies that the results are for featuresABtogether (all ML features)
        thisModelMatrix[2,j*2+0] = resultsABtogether[I][1].sensitivity[j]
        thisModelMatrix[2,j*2+1] = resultsABtogether[I][1].precision[j]
        
        # row number = 3 in "thisModelMatrix" signifies that the results are for featuresC (biomechanical)
        thisModelMatrix[3,j*2+0] = resultsC[I][1].sensitivity[j]
        thisModelMatrix[3,j*2+1] = resultsC[I][1].precision[j]
        
        # end of class-wise computation
        
    # adding the overall model accuracy at the end for each feature-set    
    thisModelMatrix[0,-1] = resultsA[I][1].accuracy
    thisModelMatrix[1,-1] = resultsB[I][1].accuracy
    thisModelMatrix[2,-1] = resultsABtogether[I][1].accuracy
    thisModelMatrix[3,-1] = resultsC[I][1].accuracy
        
    resultsMatrix.append(thisModelMatrix)
    
resultsMatrix = asarray(resultsMatrix).reshape([numFeatureSets*len(models), numClasses*2+1]) # making it a two-dimensional array


# Writing this array to a csv file, keeping just one decimal point

savetxt('table[04a1]_top3features_kFold_agg.csv', resultsMatrix, delimiter=',', fmt='%.1f')


#%% Writing mean and standard deviation of performance metrics (across all folds) to a CSV file

import csv

filename = 'table[04a1]_top3features_kFold_meanstd.csv'

for I in range(len(models)):
    
    # Writing the results for the top 3 features from feature-set A
    
    resultrowstring = [] # string containing the class-wise results for this model
    
    for j in range(numClasses):
        
        Sm = resultsA[I][2].sensitivity_mean[j] # mean sensitivity
        Sstd = resultsA[I][2].sensitivity_std[j] # std of sensitivity
        Pm = resultsA[I][2].precision_mean[j] # mean precision
        Pstd = resultsA[I][2].precision_std[j] # std of precision
        
        resultrowstring.append(('%.1f' % Sm)+' ± '+('%.1f' % Sstd))
        resultrowstring.append(('%.1f' % Pm)+' ± '+('%.1f' % Pstd))
        
        # end of class-wise computation
        
    # adding the overall model accuracy (mean ± std) at the end for feature-set A
    Accm = resultsA[I][2].accuracy_mean
    Accstd = resultsA[I][2].accuracy_std
    resultrowstring.append(('%.1f' % Accm)+' ± '+('%.1f' % Accstd))
    
    # writing this row of results for the current model to a CSV file
    with open(filename,'a',newline='') as resultFile:
        wr = csv.writer(resultFile, dialect='excel')
        wr.writerow(resultrowstring)
    
    # end of top-3 A (statistical)--------------------------------------------
    
    # Writing the results for the top 3 features from feature-set B
    
    resultrowstring = [] # string containing the class-wise results for this model
    
    for j in range(numClasses):
        
        Sm = resultsB[I][2].sensitivity_mean[j] # mean sensitivity
        Sstd = resultsB[I][2].sensitivity_std[j] # std of sensitivity
        Pm = resultsB[I][2].precision_mean[j] # mean precision
        Pstd = resultsB[I][2].precision_std[j] # std of precision
        
        resultrowstring.append(('%.1f' % Sm)+' ± '+('%.1f' % Sstd))
        resultrowstring.append(('%.1f' % Pm)+' ± '+('%.1f' % Pstd))
        
        # end of class-wise computation
        
    # adding the overall model accuracy (mean ± std) at the end for feature-set B
    Accm = resultsB[I][2].accuracy_mean
    Accstd = resultsB[I][2].accuracy_std
    resultrowstring.append(('%.1f' % Accm)+' ± '+('%.1f' % Accstd))
    
    # writing this row of results for the current model to a CSV file
    with open(filename,'a',newline='') as resultFile:
        wr = csv.writer(resultFile, dialect='excel')
        wr.writerow(resultrowstring)
    
    # end of top-3 B (human)---------------------------------------------------
    
    # Writing the results for the top 3 features from feature-set ABtogether
    
    resultrowstring = [] # string containing the class-wise results for this model
    
    for j in range(numClasses):
        
        Sm = resultsABtogether[I][2].sensitivity_mean[j] # mean sensitivity
        Sstd = resultsABtogether[I][2].sensitivity_std[j] # std of sensitivity
        Pm = resultsABtogether[I][2].precision_mean[j] # mean precision
        Pstd = resultsABtogether[I][2].precision_std[j] # std of precision
        
        resultrowstring.append(('%.1f' % Sm)+' ± '+('%.1f' % Sstd))
        resultrowstring.append(('%.1f' % Pm)+' ± '+('%.1f' % Pstd))
        
        # end of class-wise computation
        
    # adding the overall model accuracy (mean ± std) at the end for feature-set ABtogether
    Accm = resultsABtogether[I][2].accuracy_mean
    Accstd = resultsABtogether[I][2].accuracy_std
    resultrowstring.append(('%.1f' % Accm)+' ± '+('%.1f' % Accstd))
    
    # writing this row of results for the current model to a CSV file
    with open(filename,'a',newline='') as resultFile:
        wr = csv.writer(resultFile, dialect='excel')
        wr.writerow(resultrowstring)
    
    # end of top-3 ABtogether (all ML features)--------------------------------
    
    # Writing the results for the top 3 features from feature-set C2
    
    resultrowstring = [] # string containing the class-wise results for this model
    
    for j in range(numClasses):
        
        Sm = resultsC[I][2].sensitivity_mean[j] # mean sensitivity
        Sstd = resultsC[I][2].sensitivity_std[j] # std of sensitivity
        Pm = resultsC[I][2].precision_mean[j] # mean precision
        Pstd = resultsC[I][2].precision_std[j] # std of precision
        
        resultrowstring.append(('%.1f' % Sm)+' ± '+('%.1f' % Sstd))
        resultrowstring.append(('%.1f' % Pm)+' ± '+('%.1f' % Pstd))
        
        # end of class-wise computation
        
    # adding the overall model accuracy (mean ± std) at the end for feature-set B
    Accm = resultsC[I][2].accuracy_mean
    Accstd = resultsC[I][2].accuracy_std
    resultrowstring.append(('%.1f' % Accm)+' ± '+('%.1f' % Accstd))
    
    # writing this row of results for the current model to a CSV file
    with open(filename,'a',newline='') as resultFile:
        wr = csv.writer(resultFile, dialect='excel')
        wr.writerow(resultrowstring)
    
    # end of top-3 C2 (biomechanical)------------------------------------------