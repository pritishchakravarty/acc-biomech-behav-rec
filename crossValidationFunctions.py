class initialiseStruct():
    pass


#%% FUNCTIONS TO CARRY OUT CROSS VALIDATION FOR OVERALL MACHINE LEARNING (as opposed to constructing a biomechanical model)

def kFoldResults(model,X,Y,numFolds=10,balanceMethod='none'):
    
    """Given a model, a feature matrix ("X"), a vector of true labels ("Y"), 
    and (optionally) the number of folds ("numFolds"), this function does 
    stratified cross validation and produces fold-wise and aggregated (across 
    folds) results.
    OUTPUTS
    "aggregatedResults" has four parameters and is computed from the confusion
    matrices of all folds, added together to form a single confusion matrix:
        confMat
        sensitivity
        specificity
        precision
        accuracy
    "meanResults" has eight parameters and is a mean across results for each 
    fold:
        confMat_mean
        confMat_std
        sensitivity_mean
        sensitivity_std
        specificity_mean
        specificity_std
        precision_mean
        precision_std
        accuracy_mean
        accuracy_std
    "foldWiseResults" has six parameters:
        testIndices (for each fold)
        predictedLabels (for each fold)
        confMat (for each fold)
        sensitivity (for each fold)
        specificity (for each fold)
        precision (for each fold)
        accuracy (for each fold)
        
    Note that to access the values in the "struct-like" (inspired by MATLAB
    usage) outputs, one needs to create separate variables to see them. For
    instance, to see the aggregated confusion matrix, one needs to type:
        acm = aggregatedResults.confMat
    Then, "acm" will contain the required confusion matrix, and can be printed,
    etc. If you directly try to do "print(aggregatedResults.confMat)", you get
    an error.
        
    INPUTS
    "model" is, for instance: "model = GaussianNB()"
    "X" is the feature matrix, with rows corresponding to observations, and 
    columns to features
    "Y" is the list of labels for the observations in "X". Its length needs to 
    be exactly equal to the number of rows in "X"
    "numFolds" is set to 10 by default, but can be changed, but has to be at
    least 2
    "balanceMethod" is set to 'none' by default (in which case classifiers will
    be built on potentially imbalanced data). The only option for this variable
    is 'SMOTE' (type 'svm') and 'undersample' (randomly undersamples all big 
    classes to the minority class during training) for now.
    
    Written:
        16 Aug, 2017
        Pritish Chakravarty
        LMAM, EPFL
        
    Updates:
        [5 Oct, 2017] Added 'undersample' for "balanceMethod"
    """
    #==========================================================================
    
    # IMPORTING THE REQUIRED LIBRARIES
    from numpy import size
    from numpy import unique
    from numpy import zeros
    from numpy import divide
    from numpy import mean
    from numpy import std
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import confusion_matrix
    from sklearn.base import clone
    
    if balanceMethod is 'SMOTE':
        from imblearn.over_sampling import SMOTE
#==============================================================================
#         from collections import Counter
#==============================================================================
        
    elif balanceMethod is 'undersample':
        from imblearn.under_sampling import RandomUnderSampler
    
    #==========================================================================
    
    # find a list of the (numerically coded) class labels, arranged in ascending order
    classLabels = sorted(unique(Y)) # "sorted" sorts in ascending order by default
    
    # find number of classes from list of true labels
    numClasses = size(classLabels) # automatically finds number of classes [we use this to define the size of the confusion matrix later]
    
    # generating stratified folds
    seed = 100 #so that the fold generation is repeatable across different runs of the various ML algos
    skf = StratifiedKFold(n_splits=numFolds,shuffle=False,random_state=seed)
    foldIndices = list(skf.split(X,Y)) # Note that the "training" indices come first, and "testing" indices come second
    
    #==========================================================================
    
    # ------INITIALISATION OF VARIABLES THAT WILL BE STORED FOR EACH FOLD------
    
    # initialising the test indices for each fold
    testIndices = []
    # predicted labels for each fold
    predictedLabels = []
    # initialising the confusion matrix for each fold
    confMat = []
    # initialising the sensitivity for each fold (it will contain the one-V/S-all sensitivity for each class)
    sensitivity_eachFold = [zeros([1,numClasses])[0] for i in range(numFolds)]
    # initialising the specificty for each fold (it will contain the one-V/S-all specificity for each class)
    specificity_eachFold = [zeros([1,numClasses])[0] for i in range(numFolds)]
    # initialising the precision for each fold (it will contain the one-V/S-all precision for each class)
    precision_eachFold = [zeros([1,numClasses])[0] for i in range(numFolds)]
    # initialising the overall accuracy for each fold (equal to the sum of the diagonal elements of the confusion matrix divided by the sum of all the elements of the confusion matrix)
    accuracy_eachFold = zeros([numFolds,])
    # initialising a list that will store the fitted model that was computed for each fold (from which each model's parameters may later be accessed)
    trainedModel_eachFold = []
    
    #==========================================================================
    
    # ------------TRAINING THE MODEL AND TESTING IT FOR EACH FOLD--------------
    
    foldCount = 0 # counts the current "fold number" in the loop below
    

    for train_idx, test_idx in foldIndices:      # For each fold
                
        m = clone(model) # using "clone" to ensure that the model for each iteration is "brand-new" (and doesn't re-use the parameters computed in the last iteration!)
        
        # DO CLASS-BALANCING IF NEEDED, AND TRAIN THE MODEL ON BALANCED TRAINING DATA
        if balanceMethod is 'none':
            # don't change the class distributions in this case
            
            # train the model cloned in this iteration and store its parameters in one go
            trainedModel_eachFold.append(m.fit(X[train_idx],Y[train_idx]))
            
        elif balanceMethod is 'SMOTE':
            # oversample using the function "SMOTE" in imbalanced-learn so that all classes are balanced
            X_resampled, Y_resampled = SMOTE(kind='svm').fit_sample(X[train_idx],Y[train_idx])
#==============================================================================
#             # check if data has been resampled
#             print(sorted(Counter(Y_resampled).items()))
#==============================================================================
            
            # now train the model on this oversampled, class-balanced data, and save it in one go
            trainedModel_eachFold.append(m.fit(X_resampled,Y_resampled))
            
        elif balanceMethod is 'undersample':
            # undersample majority classes to minority class using the function "RandomUnderSampler", so that all classes are balanced
            X_resampled, Y_resampled = RandomUnderSampler(random_state=0).fit_sample(X[train_idx],Y[train_idx])
#==============================================================================
#             # check if data has been resampled
#             print(sorted(Counter(Y_resampled).items()))
#==============================================================================
            
            # now train the model on this oversampled, class-balanced data, and save it in one go
            trainedModel_eachFold.append(m.fit(X_resampled,Y_resampled))
            
        # END of class-balancing (if needed) and model-training
            
        
        # using the model trained (fit) above, do predictions on the test set (in other words, test the model that was trained in this iteration on the current test fold)
        Y_predicted = m.predict(X[test_idx])
        
        # compute confusion matrix calculated for test data in this iteration
        Y_true = Y[test_idx]
        C = confusion_matrix(Y_true,Y_predicted,labels=classLabels) # "true" classes along the rows and "predicted" classes along the columns
        confMat.append(C)
        
        # saving the test indices for this fold
        testIndices.append(test_idx)
        
        # saving the predicted labels for the test data used in this fold
        predictedLabels.append(Y_predicted)
        
        # calculating performance metrics for each class for each fold
        sensitivity_eachFold[foldCount], specificity_eachFold[foldCount], precision_eachFold[foldCount], accuracy_eachFold[foldCount] = computePerformanceMetrics(C)
        
        foldCount += 1 # updating "foldCount" before entering the iteration for the next fold
        
        # end of all fold-wise calculations
        
    #==========================================================================
        
    # storing all fold-wise variables in "foldWiseResults"
    foldWiseResults = initialiseStruct()
    foldWiseResults.testIndices = testIndices
    foldWiseResults.predictedLabels = predictedLabels
    foldWiseResults.confMat = confMat
    foldWiseResults.sensitivity = sensitivity_eachFold
    foldWiseResults.specificity = specificity_eachFold
    foldWiseResults.precision = precision_eachFold
    foldWiseResults.accuracy = accuracy_eachFold
    foldWiseResults.trainedModel = trainedModel_eachFold
    
    #==========================================================================
    
    # ------CALCULATING AGGREGATED (across all folds) PERFORMANCE METRICS------
    
    # calculating the aggregated confusion matrix
    confMat_allFolds = zeros([numClasses,numClasses])
    for i in range(numFolds):
        confMat_allFolds += confMat[i]
        
    # calculating aggregated performance metrics from the aggregated confusion matrix
    sensitivity_allFolds, specificity_allFolds, precision_allFolds, accuracy_allFolds = computePerformanceMetrics(confMat_allFolds)
    
    # storing all aggregated performance metrics in "aggregatedResults"
    aggregatedResults = initialiseStruct()
    aggregatedResults.confMat = confMat_allFolds
    aggregatedResults.sensitivity = sensitivity_allFolds
    aggregatedResults.specificity = specificity_allFolds
    aggregatedResults.precision = precision_allFolds
    aggregatedResults.accuracy = accuracy_allFolds
    
    #==========================================================================
    
    # -----------------CALCULATING MEAN PERFORMANCE METRICS--------------------
    
    # calculating the mean confusion matrix (across all folds)
    confMat_mean = divide(aggregatedResults.confMat,numFolds)
    # calculating the standard deviation of each respective entry in the confusion matrix (across all folds)
    confMat_std = std(foldWiseResults.confMat,axis=0,ddof=1)
    """ "axis" means that the corresponding elements across all the matrices 
    (all the [1,2] elements in each matrix, say) are assembled in a vector, and
    the std is calculated for this vector. "ddof" specifies the denominator of
    the std expression, given as N-ddof"""
    
    # calculating the mean sensitivity (across all folds)
    sensitivity_mean = mean(foldWiseResults.sensitivity,axis=0)
    # calculating the standard deviation of sensitivity (across all folds)
    sensitivity_std = std(foldWiseResults.sensitivity,axis=0,ddof=1)
    
    # calculating the mean specificity (across all folds)
    specificity_mean = mean(foldWiseResults.specificity,axis=0)
    # calculating the standard deviation of specificity (across all folds)
    specificity_std = std(foldWiseResults.specificity,axis=0,ddof=1)

    # calculating the mean precision (across all folds)
    precision_mean = mean(foldWiseResults.precision,axis=0)
    # calculating the standard deviation of precision (across all folds)
    precision_std = std(foldWiseResults.precision,axis=0,ddof=1)
    
    # calculating the mean accuracy (across all folds)
    accuracy_mean = mean(foldWiseResults.accuracy,axis=0)
    # calculating the standard deviation of accuracy (across all folds)
    accuracy_std = std(foldWiseResults.accuracy,axis=0,ddof=1)  
    
    
    meanResults = initialiseStruct()
    meanResults.confMat_mean = confMat_mean
    meanResults.confMat_std = confMat_std
    meanResults.sensitivity_mean = sensitivity_mean
    meanResults.sensitivity_std = sensitivity_std
    meanResults.specificity_mean = specificity_mean
    meanResults.specificity_std = specificity_std
    meanResults.precision_mean = precision_mean
    meanResults.precision_std = precision_std
    meanResults.accuracy_mean = accuracy_mean
    meanResults.accuracy_std = accuracy_std
    
    #==========================================================================
    
    # finally, return all the three types of results
    return aggregatedResults, meanResults, foldWiseResults
    
    
#==============================================================================
# # test
# model = GaussianNB()
# aggregatedResults, foldWiseResults = getCrossValidationResults(model,X,Y,numFolds=10)
#==============================================================================


def leaveOneMeerkatOutResults(model,X,Y,N,balanceMethod='none'):
    
    """Given a model, a feature matrix ("X"), a vector of true labels ("Y"), 
    and a vector containing the meerkat (or, in generally, subject) number,
    this function trains a model on n-1 meerkats (or subjects), where n is the
    number of distinct meerkats (or subjects), and then tests it on the
    remaining meerkat (or subject). It sequentially leaves out one meerkat
    after another, and compiles the meerkat-wise model performance (i.e. the
    results for the "first meerkat" (i.e. n=1) will be when the model was 
    trained for all the remaining meerkats, and tested on the meerkat with 
    number specified as 1), and the aggregated (across meerkats) model
    performance.
    OUTPUTS
    "aggregatedResults" has four parameters and is computed from the confusion
    matrices of all meerkats,, added together to form a single confusion matrix:
        confMat
        sensitivity
        specificity
        precision
        accuracy
    "meanResults" has eight parameters and is a mean across results for each 
    meerkat:
        confMat_mean
        confMat_std
        sensitivity_mean
        sensitivity_std
        specificity_mean
        specificity_std
        precision_mean
        precision_std
        accuracy_mean
        accuracy_std
    "meerkatWiseResults" has seven parameters:
        testIndices (for each meerkat)
        predictedLabels (for each meerkat)
        confMat (for each meerkat)
        sensitivity (for each meerkat)
        specificity (for each meerkat)
        precision (for each meerkat)
        accuracy (for each meerkat)
        trainedModel (for each meerkat)
        
    Note that to access the values in the "struct-like" (inspired by MATLAB
    usage) outputs, one needs to create separate variables to see them. For
    instance, to see the aggregated confusion matrix, one needs to type:
        acm = aggregatedResults.confMat
    Then, "acm" will contain the required confusion matrix, and can be printed,
    etc. If you directly try to do "print(aggregatedResults.confMat)", you get
    an error.
        
    INPUTS
    "model" is, for instance: "model = GaussianNB()"
    "X" is the feature matrix, with rows corresponding to observations, and 
    columns to features
    "Y" is the list of labels for the observations in "X". Its length needs to 
    be exactly equal to the number of rows in "X"
    "N" is the list of meerkat numbers. It identifies which meerkat each row in
    the feature matrix "X" corresponds to, and thus its length needs to be
    equal to the number of rows in "X"
    "balanceMethod":
        set to 'none' by default (in which case classifiers will be built on 
        potentially imbalanced data). The only option for this variable is 
        'SMOTE' (type 'svm') and 'undersample' (randomly undersamples all big 
        classes to the minority class during training) for now.
    
    Written:
        18 Aug, 2017
        Pritish Chakravarty
        LMAM, EPFL
        
    Updates:
        [06 Oct, 2017] Added "balanceMethod" for 'SMOTE' and 'undersample'
    """
    #==========================================================================
    
    # IMPORTING THE REQUIRED LIBRARIES
    from numpy import size
    from numpy import unique
    from numpy import zeros
    from numpy import divide
    from numpy import mean
    from numpy import std
    from numpy import array
    
    from sklearn.metrics import confusion_matrix
    from sklearn.base import clone
    
    if balanceMethod is 'SMOTE':
        from imblearn.over_sampling import SMOTE
#==============================================================================
#         from collections import Counter
#==============================================================================
        
    elif balanceMethod is 'undersample':
        from imblearn.under_sampling import RandomUnderSampler
    
    #==========================================================================
    
    # find a list of the (numerically coded) class labels, arranged in ascending order
    classLabels = sorted(unique(Y)) # "sorted" sorts in ascending order by default
    
    # find number of classes from list of true labels
    numClasses = size(classLabels) # automatically finds number of classes [we use this to define the size of the confusion matrix later]
    
    # find number of meerkats from list of meerkat numbers
    numMeerkats = size(unique(N)) # automatically finds number of meerkats for which data was collected [we shall use this to determine how many pairs of training and testing indices need to be generated]
    
    #==========================================================================
    
    # ----INITIALISATION OF VARIABLES THAT WILL BE STORED FOR EACH MEERKAT-----
    
    # test indices for each meerkat
    testIndices = []
    # predicted labels for each meerkat
    predictedLabels = []
    # initialising the confusion matrix for each meerkat
    confMat = []
    # initialising the sensitivity for each meerkat (it will contain the one-V/S-all sensitivity for each class)
    sensitivity_eachMeerkat = [zeros([1,numClasses])[0] for i in range(numMeerkats)]
    # initialising the specificty for each meerkat (it will contain the one-V/S-all specificity for each class)
    specificity_eachMeerkat = [zeros([1,numClasses])[0] for i in range(numMeerkats)]
    # initialising the precision for each meerkat (it will contain the one-V/S-all precision for each class)
    precision_eachMeerkat = [zeros([1,numClasses])[0] for i in range(numMeerkats)]
    # initialising the overall accuracy for each meerkat (equal to the sum of the diagonal elements of the confusion matrix divided by the sum of all the elements of the confusion matrix)
    accuracy_eachMeerkat = zeros([numMeerkats,])
    # initialising a list that will store the fitted model that was computed for each meerkat (from which each model's parameters may later be accessed)
    trainedModel_eachMeerkat = []
    
    #==========================================================================
    
    # ----------TRAINING THE MODEL AND TESTING IT FOR EACH MEERKAT-------------
    
    count = 0 # counts the number of iterations of the loop below
    
    for i in unique(N):      # For each meerkat
        
        # calculating the training and testing indices when the i-th meerkat is left out of the training phase
        """ When the i-th meerkat is left out, the testing indices will be those 
        rows that correpond to N==i, and the training indices will be all the other
        rows (i.e. for which N!=i) """
        test_idx = array([idx for idx in range(len(N)) if N[idx]==i])
        train_idx = array([idx for idx in range(len(N)) if N[idx]!=i])
            
        m = clone(model) # using "clone" to ensure that the model for each iteration is "brand-new" (and doesn't re-use the parameters computed in the last iteration!)
        
        # DO CLASS-BALANCING IF NEEDED, AND TRAIN THE MODEL ON BALANCED TRAINING DATA
        if balanceMethod is 'none':
            # don't change the class distributions in this case
            
            # train the model cloned in this iteration and store its parameters in one go
            trainedModel_eachMeerkat.append(m.fit(X[train_idx],Y[train_idx]))
            
        elif balanceMethod is 'SMOTE':
            # oversample using the function "SMOTE" in imbalanced-learn so that all classes are balanced
            X_resampled, Y_resampled = SMOTE(kind='svm').fit_sample(X[train_idx],Y[train_idx])
#==============================================================================
#             # check if data has been resampled
#             print(sorted(Counter(Y_resampled).items()))
#==============================================================================
            
            # now train the model on this oversampled, class-balanced data, and save it in one go
            trainedModel_eachMeerkat.append(m.fit(X_resampled,Y_resampled))
            
        elif balanceMethod is 'undersample':
            # undersample majority classes to minority class using the function "RandomUnderSampler", so that all classes are balanced
            X_resampled, Y_resampled = RandomUnderSampler(random_state=0).fit_sample(X[train_idx],Y[train_idx])
#==============================================================================
#             # check if data has been resampled
#             print(sorted(Counter(Y_resampled).items()))
#==============================================================================
            
            # now train the model on this oversampled, class-balanced data, and save it in one go
            trainedModel_eachMeerkat.append(m.fit(X_resampled,Y_resampled))
            
        # END of class-balancing (if needed) and model-training
        
        # test the model on the meerkat's data that was left out during the training phase
        Y_predicted = m.predict(X[test_idx])
        
        # compute confusion matrix calculated for test data in this iteration
        Y_true = Y[test_idx]
        C = confusion_matrix(Y_true,Y_predicted,labels=classLabels) # "true" classes along the rows and "predicted" classes along the columns
        confMat.append(C)
        
        # saving the test indices for this iteration (i.e. the rows in "X" that correspond to the current meerkat)
        testIndices.append(test_idx)
        
        # saving the predicted labels for this meerkat's data
        predictedLabels.append(Y_predicted)
        
        # calculating performance metrics for each class for this meerkat's data
        sensitivity_eachMeerkat[count], specificity_eachMeerkat[count], precision_eachMeerkat[count], accuracy_eachMeerkat[count] = computePerformanceMetrics(C)
        
        count += 1
        
        # END of all meerkat-wise calculations
        
    #==========================================================================
        
    # storing all meerkat-wise variables in "meerkatWiseResults"
    meerkatWiseResults = initialiseStruct()
    meerkatWiseResults.testIndices = testIndices
    meerkatWiseResults.predictedLabels = predictedLabels
    meerkatWiseResults.confMat = confMat
    meerkatWiseResults.sensitivity = sensitivity_eachMeerkat
    meerkatWiseResults.specificity = specificity_eachMeerkat
    meerkatWiseResults.precision = precision_eachMeerkat
    meerkatWiseResults.accuracy = accuracy_eachMeerkat
    meerkatWiseResults.trainedModel = trainedModel_eachMeerkat
    
    #==========================================================================
    
    # -----CALCULATING AGGREGATED (across all meerkats) PERFORMANCE METRICS----
    
    # calculating the aggregated confusion matrix
    confMat_allMeerkats = zeros([numClasses,numClasses])
    for i in range(numMeerkats):
        confMat_allMeerkats += confMat[i]
        
    # calculating aggregated performance metrics from the aggregated confusion matrix
    sensitivity_allMeerkats, specificity_allMeerkats, precision_allMeerkats, accuracy_allMeerkats = computePerformanceMetrics(confMat_allMeerkats)
    
    # storing all aggregated performance metrics in "aggregatedResults"
    aggregatedResults = initialiseStruct()
    aggregatedResults.confMat = confMat_allMeerkats
    aggregatedResults.sensitivity = sensitivity_allMeerkats
    aggregatedResults.specificity = specificity_allMeerkats
    aggregatedResults.precision = precision_allMeerkats
    aggregatedResults.accuracy = accuracy_allMeerkats
    
    #==========================================================================
    
    # -----------------CALCULATING MEAN PERFORMANCE METRICS--------------------
    
    # calculating the mean confusion matrix (across all meerkats)
    confMat_mean = divide(aggregatedResults.confMat,numMeerkats)
    # calculating the standard deviation of each respective entry in the confusion matrix (across all meerkats)
    confMat_std = std(meerkatWiseResults.confMat,axis=0,ddof=1)
    """ "axis" means that the corresponding elements across all the matrices 
    (all the [1,2] elements in each matrix, say) are assembled in a vector, and
    the std is calculated for this vector. "ddof" specifies the denominator of
    the std expression, given as N-ddof"""
    
    # calculating the mean sensitivity (across all meerkats)
    sensitivity_mean = mean(meerkatWiseResults.sensitivity,axis=0)
    # calculating the standard deviation of sensitivity (across all meerkats)
    sensitivity_std = std(meerkatWiseResults.sensitivity,axis=0,ddof=1)
    
    # calculating the mean specificity (across all meerkats)
    specificity_mean = mean(meerkatWiseResults.specificity,axis=0)
    # calculating the standard deviation of specificity (across all meerkats)
    specificity_std = std(meerkatWiseResults.specificity,axis=0,ddof=1)

    # calculating the mean precision (across all meerkats)
    precision_mean = mean(meerkatWiseResults.precision,axis=0)
    # calculating the standard deviation of precision (across all meerkats)
    precision_std = std(meerkatWiseResults.precision,axis=0,ddof=1)    
    
    # calculating the mean accuracy (across all meerkats)
    accuracy_mean = mean(meerkatWiseResults.accuracy,axis=0)
    # calculating the standard deviation of accuracy (across all meerkats)
    accuracy_std = std(meerkatWiseResults.accuracy,axis=0,ddof=1)  
    
    
    meanResults = initialiseStruct()
    meanResults.confMat_mean = confMat_mean
    meanResults.confMat_std = confMat_std
    meanResults.sensitivity_mean = sensitivity_mean
    meanResults.sensitivity_std = sensitivity_std
    meanResults.specificity_mean = specificity_mean
    meanResults.specificity_std = specificity_std
    meanResults.precision_mean = precision_mean
    meanResults.precision_std = precision_std
    meanResults.accuracy_mean = accuracy_mean
    meanResults.accuracy_std = accuracy_std
    
    #==========================================================================
    
    
    # finally, return all the three types of results
    return aggregatedResults, meanResults, meerkatWiseResults
        

#%% FUNCTIONS TO COMPUTE PERFORMANCE METRICS FOR THE BIOMECHANICAL MODELS

def kFoldResults_biomechanicalModel(threeModels,featureIdx,X,Y,Y_SD,numFolds=10,balanceMethod='none'):
    
    """Given a set of three models and the features to be used for each model,
    this function does stratified cross validation and produces fold-wise and 
    aggregated (across folds) results for the final 4-class confusion matrices
    (i.e. not the intermediate static vs dynamic results).
    
    INPUTS
    
    "threeModels":
        A list of three machine learning models. The first model will do 
        static versus dynamic classification, the second will do vigilance versus
        resting classification, and the third will do foraging versus running
        classification. Each of the three entries of "threeModels" must be a list
        of two entries which are, in sequence: a short string which will serve
        as the name of the model, and then the model itself. An example entry
        could be, for instance: threeModels[0] = ('NB', GaussianNB()). This 
        contains a Gaussian Naïve-Bayes model and a short form ("NB") to refer
        to it
    "featureIdx":
        A list of indices of three feature sets that will be used, in sequence, for the 
        models in "threeModels". This is a list of indices referring to column
        numbers in "X". For instance, if "X" has 5 columns, this input can be
        of the form: [[1,2], 1, [2,3,4]], where 1:meanX, 2:stdNorm, 3:fftPeakPowerAvg,
        4:deviationFromFreefall
    "X":
        Feature matrix, with rows corresponding to observations and columns to
        features
    "Y":
        Behaviour labels derived from videos. In this case, there are just 4 
        behaviour labels - 1:vigilance, 2:resting, 3:foraging, 4:running
    "Y_SD":
        Behaviour labels grouped into "static" (labelled as 1) and "dynamic"
        (labelled as 2). Since each element in "Y" and "Y_SD" corresponds to 
        one observation (i.e. window of acceleration corresponding to one
        particular class), the number of rows in "X", "Y" and "Y_SD" should be
        exactly the same
    "numFolds":
        Optional variable, denotes the value of "k" in k-fold cross-validation.
        Set to 10 by default. Has to be at least 2
    "balanceMethod":
        set to 'none' by default (in which case classifiers will be built on 
        potentially imbalanced data). The only option for this variable is 
        'SMOTE' (type 'svm') and 'undersample' (randomly undersamples all big 
        classes to the minority class during training) for now.
    
    
    OUTPUTS
    
    "aggregatedResults" has four parameters and is computed from the confusion
    matrices of all folds, added together to form a single confusion matrix:
        confMat
        sensitivity
        specificity
        precision
        accuracy
    "meanResults" has eight parameters and is a mean across results for each 
    fold:
        confMat_mean
        confMat_std
        sensitivity_mean
        sensitivity_std
        specificity_mean
        specificity_std
        precision_mean
        precision_std
        accuracy_mean
        accuracy_std
    "foldWiseResults" has six parameters:
        testIndices (for each fold)
        predictedLabels (for each fold)
        confMat (for each fold)
        sensitivity (for each fold)
        specificity (for each fold)
        precision (for each fold)
        accuracy (for each fold)
        
    Note that to access the values in the "struct-like" (inspired by MATLAB
    usage) outputs, one needs to create separate variables to see them. For
    instance, to see the aggregated confusion matrix, one needs to type:
        acm = aggregatedResults.confMat
    Then, "acm" will contain the required confusion matrix, and can be printed,
    etc. If you directly try to do "print(aggregatedResults.confMat)", you get
    an error.
    
    Written:
        03 Sep, 2017
        Pritish Chakravarty
        LMAM, EPFL
        
    Updates:
        [05 Oct, 2017] Added "balanceMethod" for SMOTE and 'undersample'
    """
    #==========================================================================
    
    # IMPORTING THE REQUIRED LIBRARIES
    from numpy import size
    from numpy import unique
    from numpy import zeros
    from numpy import divide
    from numpy import mean
    from numpy import std
    from numpy import append
    from numpy import array
    
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import confusion_matrix
    from sklearn.base import clone
    
    if balanceMethod is 'SMOTE':
        from imblearn.over_sampling import SMOTE
#==============================================================================
#         from collections import Counter
#==============================================================================
        
    elif balanceMethod is 'undersample':
        from imblearn.under_sampling import RandomUnderSampler
    
    #==========================================================================
    
    # find a list of the (numerically coded) class labels, arranged in ascending order
    classLabels = sorted(unique(Y)) # "sorted" sorts in ascending order by default
    
    # find number of classes from list of true labels
    numClasses = size(classLabels) # automatically finds number of classes [we use this to define the size of the confusion matrix later]
    
    # generating stratified folds
    seed = 100 # so that the fold generation is repeatable across different runs of the various ML algos
    skf = StratifiedKFold(n_splits=numFolds,shuffle=False,random_state=seed)
    foldIndices = list(skf.split(X,Y)) # Note that the "training" indices come first, and "testing" indices come second
    
    #==========================================================================
    
    # ------INITIALISATION OF VARIABLES THAT WILL BE STORED FOR EACH FOLD------
    
    # initialising the test indices for each fold
    testIndices = []
    # predicted labels for each fold
    predictedLabels = []
    # initialising the confusion matrix for each fold
    confMat = []
    # initialising the sensitivity for each fold (it will contain the one-V/S-all sensitivity for each class)
    sensitivity_eachFold = [zeros([1,numClasses])[0] for i in range(numFolds)]
    # initialising the specificty for each fold (it will contain the one-V/S-all specificity for each class)
    specificity_eachFold = [zeros([1,numClasses])[0] for i in range(numFolds)]
    # initialising the precision for each fold (it will contain the one-V/S-all precision for each class)
    precision_eachFold = [zeros([1,numClasses])[0] for i in range(numFolds)]
    # initialising the overall accuracy for each fold (sum of diagonal elements of the confusion matrix divided by the sum of all elements of the confusion matrix)
    accuracy_eachFold = zeros([numFolds,])
    # initialising lists that will store the fitted model that was computed for each fold (from which each model's parameters may later be accessed)
    trainedModel_staticDynamic_eachFold = [] # model trained for static versus dynamic activity classification for each fold
    trainedModel_vigRest_eachFold = [] # model trained for vigilance versus resting classification for each fold
    trainedModel_foragRun_eachFold = [] # model trained for foraging versus running classification for each fold
    
    #==========================================================================
    
    # ------------TRAINING THE MODEL AND TESTING IT FOR EACH FOLD--------------
    
    foldCount = 0 # counts the current "fold number" in the loop below

    for train_idx, test_idx in foldIndices:      # For each fold
    
        #----------------------------------------------------------------------
        
        # DOING STATIC VERSUS DYNAMIC CLASSIFICATION
        
        M1 = clone(threeModels[0][1]) # creating a new model (for this iteration) for static versus dynamic classification. We use "clone" to ensure that the model for each iteration is "brand-new" (and doesn't re-use the parameters computed in the last iteration!)
        
        X_SD = X[:,featureIdx[0]] # only the features in "featureIdx[0] will be used for static VS dynamic classification
        
        # Python doesn't like feature matrices to have just one feature (one column, i.e.). It asks one to reshape the feature matrix to FeatureMatrix.reshape(-1,1)
        if size(featureIdx[0])==1:
            
            # DO CLASS-BALANCING IF NEEDED, AND TRAIN THE MODEL ON BALANCED TRAINING DATA
            if balanceMethod is 'none':
                # don't change the class distributions in this case
                # train the static versus dynamic model and store its parameters in one go
                trainedModel_staticDynamic_eachFold.append(M1.fit(X_SD[train_idx].reshape(-1,1),Y_SD[train_idx]))
                
            elif balanceMethod is 'SMOTE':
                # oversample using the function "SMOTE" in imbalanced-learn so that all classes are balanced
                X_SD_resampled, Y_SD_resampled = SMOTE(kind='svm').fit_sample(X_SD[train_idx].reshape(-1,1),Y_SD[train_idx])
    #==============================================================================
    #             # check if data has been resampled
    #             print(sorted(Counter(Y_SD_resampled).items()))
    #==============================================================================
                # train the static versus dynamic model and store its parameters in one go
                trainedModel_staticDynamic_eachFold.append(M1.fit(X_SD_resampled.reshape(-1,1),Y_SD_resampled))
                
            elif balanceMethod is 'undersample':
                # undersample majority classes to minority class using the function "RandomUnderSampler", so that all classes are balanced
                X_SD_resampled, Y_SD_resampled = RandomUnderSampler(random_state=0).fit_sample(X_SD[train_idx].reshape(-1,1),Y_SD[train_idx])
    #==============================================================================
    #             # check if data has been resampled
    #             print(sorted(Counter(Y_SD_resampled).items()))
    #==============================================================================
                # train the static versus dynamic model and store its parameters in one go
                trainedModel_staticDynamic_eachFold.append(M1.fit(X_SD_resampled.reshape(-1,1),Y_SD_resampled))
                
            # END of class-balancing (if needed) and model-training
                
            # TESTING THE MODEL TRAINED JUST ABOVE
            # using the model trained (fit) above, do predictions on the test set (in other words, test the model that was trained in this iteration on the current test fold for static versus dynamic activity classification)
            Y_SD_predicted = M1.predict(X_SD[test_idx].reshape(-1,1)) # predicted static and dynamic labels
             
            #=================================================================
            
        else:
            
            # DO CLASS-BALANCING IF NEEDED, AND TRAIN THE MODEL ON BALANCED TRAINING DATA
            if balanceMethod is 'none':
                # don't change the class distributions in this case
                # train the static versus dynamic model and store its parameters in one go
                trainedModel_staticDynamic_eachFold.append(M1.fit(X_SD[train_idx],Y_SD[train_idx]))
                
            elif balanceMethod is 'SMOTE':
                # oversample using the function "SMOTE" in imbalanced-learn so that all classes are balanced
                X_SD_resampled, Y_SD_resampled = SMOTE(kind='svm').fit_sample(X_SD[train_idx],Y_SD[train_idx])
    #==============================================================================
    #             # check if data has been resampled
    #             print(sorted(Counter(Y_SD_resampled).items()))
    #==============================================================================
                # train the static versus dynamic model and store its parameters in one go
                trainedModel_staticDynamic_eachFold.append(M1.fit(X_SD_resampled,Y_SD_resampled))
                
            elif balanceMethod is 'undersample':
                # undersample majority classes to minority class using the function "RandomUnderSampler", so that all classes are balanced
                X_SD_resampled, Y_SD_resampled = RandomUnderSampler(random_state=0).fit_sample(X_SD[train_idx],Y_SD[train_idx])
    #==============================================================================
    #             # check if data has been resampled
    #             print(sorted(Counter(Y_SD_resampled).items()))
    #==============================================================================
                # train the static versus dynamic model and store its parameters in one go
                trainedModel_staticDynamic_eachFold.append(M1.fit(X_SD_resampled,Y_SD_resampled))
                
                # END of class-balancing (if needed) and model-training
                
            # TESTING THE MODEL TRAINED JUST ABOVE
            # using the model trained (fit) above, do predictions on the test set (in other words, test the model that was trained in this iteration on the current test fold for static versus dynamic activity classification)
            Y_SD_predicted = M1.predict(X_SD[test_idx]) # predicted static and dynamic labels
             
            # END of static-dynamic model-training and prediction on test-set
             
            #==================================================================

            
        
        # Y_SD_true = Y_SD[test_idx] # true (from video-labels) static and dynamic labels
        
        #----------------------------------------------------------------------
        
        # DOING VIGILANCE VERSUS RESTING CLASSIFICATION
        
        M2 = clone(threeModels[1][1]) # creating a new model (for this iteration) for vigilance versus resting classification
        
        X_VRst = X[:,featureIdx[1]] # only the features in "featureIdx[1] will be used for vigilance VS resting classification
                
        # in training data, choose only those labels which correspond to vigilance and resting (which are both static activities)
        vigRest_trainIdx = [idx for idx in train_idx if Y_SD[idx]==1]
        
        # find test indices. The vig V/S rest model will be tested only on those instances which were classified as static by "M1"
        vigRest_testIdx = test_idx[[i for i in range(len(Y_SD_predicted)) if Y_SD_predicted[i]==1]]
        
        if len(vigRest_testIdx)!=0:     # it's quite possible that M1 classified all the test instances as "dynamic", and so there are none left to be classified as vigilance or rest. So, to avoid throwing up an error in the "M2.predict" line, check for this here
        
            # Python doesn't like feature matrices to have just one feature (one column, i.e.). It asks one to reshape the feature matrix to FeatureMatrix.reshape(-1,1)
            if size(featureIdx[1])==1:

                # DO CLASS-BALANCING IF NEEDED, AND TRAIN THE MODEL ON BALANCED TRAINING DATA
                if balanceMethod is 'none':
                    # don't change the class distributions in this case
                    # train the vigilance versus resting model and store its parameters in one go
                    trainedModel_vigRest_eachFold.append(M2.fit(X_VRst[vigRest_trainIdx].reshape(-1,1),Y[vigRest_trainIdx]))
                    
                elif balanceMethod is 'SMOTE':
                    # oversample using the function "SMOTE" in imbalanced-learn so that all classes are balanced
                    X_VRst_resampled, Y_VRst_resampled = SMOTE(kind='svm').fit_sample(X_VRst[vigRest_trainIdx].reshape(-1,1),Y[vigRest_trainIdx])
        #==============================================================================
        #             # check if data has been resampled
        #             print(sorted(Counter(Y_SD_resampled).items()))
        #==============================================================================
                    # train the vigilance versus resting model and store its parameters in one go
                    trainedModel_vigRest_eachFold.append(M2.fit(X_VRst_resampled.reshape(-1,1),Y_VRst_resampled))
                    
                elif balanceMethod is 'undersample':
                    # undersample majority classes to minority class using the function "RandomUnderSampler", so that all classes are balanced
                    X_VRst_resampled, Y_VRst_resampled = RandomUnderSampler(random_state=0).fit_sample(X_VRst[vigRest_trainIdx].reshape(-1,1),Y[vigRest_trainIdx])
        #==============================================================================
        #             # check if data has been resampled
        #             print(sorted(Counter(Y_SD_resampled).items()))
        #==============================================================================
                    # train the vigilance versus resting model and store its parameters in one go
                    trainedModel_vigRest_eachFold.append(M2.fit(X_VRst_resampled.reshape(-1,1),Y_VRst_resampled))
                    
                # END of class-balancing (if needed) and model-training
                
                # test the vigilance VS rest model trained in this fold    
                Y_VRst_predicted = M2.predict(X_VRst[vigRest_testIdx].reshape(-1,1))
                
                #==============================================================
                
            else:                

                # DO CLASS-BALANCING IF NEEDED, AND TRAIN THE MODEL ON BALANCED TRAINING DATA
                if balanceMethod is 'none':
                    # don't change the class distributions in this case
                    # train the vigilance versus resting model and store its parameters in one go
                    trainedModel_vigRest_eachFold.append(M2.fit(X_VRst[vigRest_trainIdx],Y[vigRest_trainIdx]))
                    
                elif balanceMethod is 'SMOTE':
                    # oversample using the function "SMOTE" in imbalanced-learn so that all classes are balanced
                    X_VRst_resampled, Y_VRst_resampled = SMOTE(kind='svm').fit_sample(X_VRst[vigRest_trainIdx],Y[vigRest_trainIdx])
        #==============================================================================
        #             # check if data has been resampled
        #             print(sorted(Counter(Y_SD_resampled).items()))
        #==============================================================================
                    # train the vigilance versus resting model and store its parameters in one go
                    trainedModel_vigRest_eachFold.append(M2.fit(X_VRst_resampled,Y_VRst_resampled))
                    
                elif balanceMethod is 'undersample':
                    # undersample majority classes to minority class using the function "RandomUnderSampler", so that all classes are balanced
                    X_VRst_resampled, Y_VRst_resampled = RandomUnderSampler(random_state=0).fit_sample(X_VRst[vigRest_trainIdx],Y[vigRest_trainIdx])
        #==============================================================================
        #             # check if data has been resampled
        #             print(sorted(Counter(Y_SD_resampled).items()))
        #==============================================================================
                    # train the vigilance versus resting model and store its parameters in one go
                    trainedModel_vigRest_eachFold.append(M2.fit(X_VRst_resampled,Y_VRst_resampled))
                    
                # END of class-balancing (if needed) and model-training
                
                #==============================================================
                
                # test the vigilance VS rest model        
                Y_VRst_predicted = M2.predict(X_VRst[vigRest_testIdx])
                
            # END of vigilance-resting model-training and prediction on test-set
                
            #==================================================================
                
                
            # find the true labels for the data rows (in "X") for which predictions were made above
            Y_VRst_true = Y[vigRest_testIdx]
                
        else:
        
            Y_VRst_predicted = array([]) # empty array
            Y_VRst_true = array([]) # also an empty array
        
        
        #----------------------------------------------------------------------
        
        # DOING FORAGING VERSUS RUNNING CLASSIFICATION
        
        M3 = clone(threeModels[2][1]) # creating a new model (for this iteration) for foraging versus running classification
        
        X_FR = X[:,featureIdx[2]] # only the features in "featureIdx[2] will be used for foraging VS running classification
                
        # in training data, choose only those labels which correspond to foraging and resting (which are both dynamic activities)
        frgRun_trainIdx = [idx for idx in train_idx if Y_SD[idx]==2]
        
        # find test indices. The foraging VS running model will be tested only on those instances which were classified as dynamic by "M1"
        frgRun_testIdx = test_idx[[i for i in range(len(Y_SD_predicted)) if Y_SD_predicted[i]==2]]
        
        if len(frgRun_testIdx)!=0:     # it's quite possible that M1 classified all the test instances as "static", and so there are none left to be classified as foraging or running. So, to avoid throwing up an error in the "M3.predict" line, check for this here

            # Python doesn't like feature matrices to have just one feature (one column, i.e.). It asks one to reshape the feature matrix to FeatureMatrix.reshape(-1,1)
            if size(featureIdx[2])==1:

                # DO CLASS-BALANCING IF NEEDED, AND TRAIN THE MODEL ON BALANCED TRAINING DATA
                if balanceMethod is 'none':
                    # don't change the class distributions in this case
                    # train the foraging versus running model and store its parameters in one go
                    trainedModel_foragRun_eachFold.append(M3.fit(X_FR[frgRun_trainIdx].reshape(-1,1),Y[frgRun_trainIdx]))
                    
                elif balanceMethod is 'SMOTE':
                    # oversample using the function "SMOTE" in imbalanced-learn so that all classes are balanced
                    X_FR_resampled, Y_FR_resampled = SMOTE(kind='svm').fit_sample(X_FR[frgRun_trainIdx].reshape(-1,1),Y[frgRun_trainIdx])
        #==============================================================================
        #             # check if data has been resampled
        #             print(sorted(Counter(Y_SD_resampled).items()))
        #==============================================================================
                    # train the foraging versus running model and store its parameters in one go
                    trainedModel_foragRun_eachFold.append(M3.fit(X_FR_resampled.reshape(-1,1),Y_FR_resampled))
                    
                elif balanceMethod is 'undersample':
                    # undersample majority classes to minority class using the function "RandomUnderSampler", so that all classes are balanced
                    X_FR_resampled, Y_FR_resampled = RandomUnderSampler(random_state=0).fit_sample(X_FR[frgRun_trainIdx].reshape(-1,1),Y[frgRun_trainIdx])
        #==============================================================================
        #             # check if data has been resampled
        #             print(sorted(Counter(Y_SD_resampled).items()))
        #==============================================================================
                    # train the foraging versus running model and store its parameters in one go
                    trainedModel_foragRun_eachFold.append(M3.fit(X_FR_resampled.reshape(-1,1),Y_FR_resampled))
                    
                # END of class-balancing (if needed) and model-training
                
                # test the vigilance VS rest model trained in this fold    
                Y_FR_predicted = M3.predict(X_FR[frgRun_testIdx].reshape(-1,1))
                
                #==============================================================
                
            else:
                
                # DO CLASS-BALANCING IF NEEDED, AND TRAIN THE MODEL ON BALANCED TRAINING DATA
                if balanceMethod is 'none':
                    # don't change the class distributions in this case
                    # train the foraging versus running model and store its parameters in one go
                    trainedModel_foragRun_eachFold.append(M3.fit(X_FR[frgRun_trainIdx],Y[frgRun_trainIdx]))
                    
                elif balanceMethod is 'SMOTE':
                    # oversample using the function "SMOTE" in imbalanced-learn so that all classes are balanced
                    X_FR_resampled, Y_FR_resampled = SMOTE(kind='svm').fit_sample(X_FR[frgRun_trainIdx],Y[frgRun_trainIdx])
        #==============================================================================
        #             # check if data has been resampled
        #             print(sorted(Counter(Y_SD_resampled).items()))
        #==============================================================================
                    # train the foraging versus running model and store its parameters in one go
                    trainedModel_foragRun_eachFold.append(M3.fit(X_FR_resampled,Y_FR_resampled))
                    
                elif balanceMethod is 'undersample':
                    # undersample majority classes to minority class using the function "RandomUnderSampler", so that all classes are balanced
                    X_FR_resampled, Y_FR_resampled = RandomUnderSampler(random_state=0).fit_sample(X_FR[frgRun_trainIdx],Y[frgRun_trainIdx])
        #==============================================================================
        #             # check if data has been resampled
        #             print(sorted(Counter(Y_SD_resampled).items()))
        #==============================================================================
                    # train the foraging versus running model and store its parameters in one go
                    trainedModel_foragRun_eachFold.append(M3.fit(X_FR_resampled,Y_FR_resampled))
                    
                # END of class-balancing (if needed) and model-training
                
                # test the vigilance VS rest model trained in this fold    
                Y_FR_predicted = M3.predict(X_FR[frgRun_testIdx])
                
                #==============================================================
                
            # END of foraging-running model-training and prediction on test-set
                
            #==================================================================
    
            # find the true labels for the data rows (in "X") for which predictions were made above
            Y_FR_true = Y[frgRun_testIdx]
            
        else:
            
            Y_FR_predicted = array([]) # empty array
            Y_FR_true = array([]) # also an empty array
        
        #----------------------------------------------------------------------
        
        """ Perhaps it is worth noting at this point that the union of 
        "vigRest_testIdx" and "frgRun_testIdx" is exactly equal to "test_idx".
        This is because "Y_SD_predicted" can only contain either 1's or 2's."""
        
        #----------------------------------------------------------------------
        
        
        # COMPUTE FINAL PREDICTED LABELS AND, FROM THEM, THE PERFORMANCE METRICS
        
        # merging the "VR" and "FR" predicted and true vectors to calculate the confusion matrix
        Y_predicted = append(Y_VRst_predicted, Y_FR_predicted)
        Y_true = append(Y_VRst_true, Y_FR_true)
        
        # computing the confusion matrix calculated for the test data in this iteration
        C = confusion_matrix(Y_true,Y_predicted,labels=classLabels) # "true" classes along the rows and "predicted" classes along the columns
        confMat.append(C)
        
        # saving the test indices for this fold
        testIndices.append(append(vigRest_testIdx, frgRun_testIdx)) # saving the test indices in the same order as they've been processed above (which was, in turn, determined by the predictions of the static VS dynamic classifier)
                
        # saving the predicted labels for the test data used in this fold
        predictedLabels.append(Y_predicted)
                
        # calculating performance metrics for each class for each fold
        sensitivity_eachFold[foldCount], specificity_eachFold[foldCount], precision_eachFold[foldCount], accuracy_eachFold[foldCount] = computePerformanceMetrics(C)
                
        foldCount += 1 # updating "foldCount" before entering the iteration for the next fold                                
        
        # end of all fold-wise calculations
        
    #==========================================================================
        
    # storing all fold-wise variables in "foldWiseResults"
    foldWiseResults = initialiseStruct()
    foldWiseResults.testIndices = testIndices
    foldWiseResults.predictedLabels = predictedLabels
    foldWiseResults.confMat = confMat
    foldWiseResults.sensitivity = sensitivity_eachFold
    foldWiseResults.specificity = specificity_eachFold
    foldWiseResults.precision = precision_eachFold
    foldWiseResults.accuracy = accuracy_eachFold
    foldWiseResults.trainedModel_staticDynamic = trainedModel_staticDynamic_eachFold
    foldWiseResults.trainedModel_vigRest = trainedModel_vigRest_eachFold
    foldWiseResults.trainedModel_foragRun = trainedModel_foragRun_eachFold
    
    #==========================================================================
    
    # ------CALCULATING AGGREGATED (across all folds) PERFORMANCE METRICS------
    
    # calculating the aggregated confusion matrix
    confMat_allFolds = zeros([numClasses,numClasses])
    for i in range(numFolds):
        confMat_allFolds += confMat[i]
        
    # calculating aggregated performance metrics from the aggregated confusion matrix
    sensitivity_allFolds, specificity_allFolds, precision_allFolds, accuracy_allFolds = computePerformanceMetrics(confMat_allFolds)
    
    # storing all aggregated performance metrics in "aggregatedResults"
    aggregatedResults = initialiseStruct()
    aggregatedResults.confMat = confMat_allFolds
    aggregatedResults.sensitivity = sensitivity_allFolds
    aggregatedResults.specificity = specificity_allFolds
    aggregatedResults.precision = precision_allFolds
    aggregatedResults.accuracy = accuracy_allFolds
    
    #==========================================================================
    
    # -----------------CALCULATING MEAN PERFORMANCE METRICS--------------------
    
    # calculating the mean confusion matrix (across all folds)
    confMat_mean = divide(aggregatedResults.confMat,numFolds)
    # calculating the standard deviation of each respective entry in the confusion matrix (across all folds)
    confMat_std = std(foldWiseResults.confMat,axis=0,ddof=1)
    """ "axis" means that the corresponding elements across all the matrices 
    (all the [1,2] elements in each matrix, say) are assembled in a vector, and
    the std is calculated for this vector. "ddof" specifies the denominator of
    the std expression, given as N-ddof"""
    
    # calculating the mean sensitivity (across all folds)
    sensitivity_mean = mean(foldWiseResults.sensitivity,axis=0)
    # calculating the standard deviation of sensitivity (across all folds)
    sensitivity_std = std(foldWiseResults.sensitivity,axis=0,ddof=1)
    
    # calculating the mean specificity (across all folds)
    specificity_mean = mean(foldWiseResults.specificity,axis=0)
    # calculating the standard deviation of specificity (across all folds)
    specificity_std = std(foldWiseResults.specificity,axis=0,ddof=1)

    # calculating the mean precision (across all folds)
    precision_mean = mean(foldWiseResults.precision,axis=0)
    # calculating the standard deviation of precision (across all folds)
    precision_std = std(foldWiseResults.precision,axis=0,ddof=1)    
    
    # calculating the mean accuracy (across all folds)
    accuracy_mean = mean(foldWiseResults.accuracy,axis=0)
    # calculating the standard deviation of accuracy (across all folds)
    accuracy_std = std(foldWiseResults.accuracy,axis=0,ddof=1)    
    
    
    meanResults = initialiseStruct()
    meanResults.confMat_mean = confMat_mean
    meanResults.confMat_std = confMat_std
    meanResults.sensitivity_mean = sensitivity_mean
    meanResults.sensitivity_std = sensitivity_std
    meanResults.specificity_mean = specificity_mean
    meanResults.specificity_std = specificity_std
    meanResults.precision_mean = precision_mean
    meanResults.precision_std = precision_std
    meanResults.accuracy_mean = accuracy_mean
    meanResults.accuracy_std = accuracy_std
    
    #==========================================================================
    
    # finally, return all the three types of results
    return aggregatedResults, meanResults, foldWiseResults


#==============================================================================

def leaveOneMeerkatOutResults_biomechanicalModel(threeModels,featureIdx,X,Y,N,Y_SD,balanceMethod='none'):
    
    """Given a set of three models and the features to be used for each model,
    this function does leave-one-meerkat-out cross-validation and produces 
    meerkat-wise and aggregated (across meerkats) results for the final 4-class
    confusion matrices (note that the intermediate static vs dynamic results
    are not reported).
    
    INPUTS
    
    "threeModels":
        A list of three machine learning models. The first model will do 
        static versus dynamic classification, the second will do vigilance versus
        resting classification, and the third will do foraging versus running
        classification. Each of the three entries of "threeModels" must be a list
        of two entries which are, in sequence: a short string which will serve
        as the name of the model, and then the model itself. An example entry
        could be, for instance: threeModels[0] = ('NB', GaussianNB()). This 
        contains a Gaussian Naïve-Bayes model and a short form ("NB") to refer
        to it
    "featureIdx":
        A list of indices of three feature sets that will be used, in sequence, for the 
        models in "threeModels". This is a list of indices referring to column
        numbers in "X". For instance, if "X" has 5 columns, this input can be
        of the form: [[1,2], 1, [2,3,4]], where 1:meanX, 2:stdNorm, 3:fftPeakPowerAvg,
        4:deviationFromFreefall
    "X":
        Feature matrix, with rows corresponding to observations and columns to
        features
    "Y":
        Behaviour labels derived from videos. In this case, there are just 4 
        behaviour labels - 1:vigilance, 2:resting, 3:foraging, 4:running
    "N":
        A vector (of the same length as the number of rows in "X") containing
        the number of the meerkat from which the data for that particular row
        was recorded.
    "Y_SD":
        Behaviour labels grouped into "static" (labelled as 1) and "dynamic"
        (labelled as 2). Since each element in "Y" and "Y_SD" corresponds to 
        one observation (i.e. window of acceleration corresponding to one
        particular class), the number of rows in "X", "Y" and "Y_SD" should be
        exactly the same
    "balanceMethod":
        set to 'none' by default (in which case classifiers will be built on 
        potentially imbalanced data). The only option for this variable is 
        'SMOTE' (type 'svm') and 'undersample' (randomly undersamples all big 
        classes to the minority class during training) for now.
    
    OUTPUTS
    
    "aggregatedResults" has four parameters and is computed from the confusion
    matrices of all meerkats, added together to form a single confusion matrix:
        confMat
        sensitivity
        specificity
        precision
        accuracy
    "meanResults" has eight parameters and is a mean across results for each 
    meerkat:
        confMat_mean
        confMat_std
        sensitivity_mean
        sensitivity_std
        specificity_mean
        specificity_std
        precision_mean
        precision_std
        accuracy_mean
        accuracy_std
    "meerkatWiseResults" has six parameters:
        testIndices (for each meerkat)
        predictedLabels (for each meerkat)
        confMat (for each meerkat)
        sensitivity (for each meerkat)
        specificity (for each meerkat)
        precision (for each meerkat)
        accuracy (for each meerkat)
        
    Note that to access the values in the "struct-like" (inspired by MATLAB
    usage) outputs, one needs to create separate variables to see them. For
    instance, to see the aggregated confusion matrix, one needs to type:
        acm = aggregatedResults.confMat
    Then, "acm" will contain the required confusion matrix, and can be printed,
    etc. If you directly try to do "print(aggregatedResults.confMat)", you get
    an error.
    
    Written:
        09 Sep, 2017
        Pritish Chakravarty
        LMAM, EPFL
        
    Updates:
        [06 Oct, 2017] Added "balanceMethod" for 'SMOTE' and 'undersample'
    """
    #==========================================================================
    
    # IMPORTING THE REQUIRED LIBRARIES
    from numpy import size
    from numpy import unique
    from numpy import zeros
    from numpy import divide
    from numpy import mean
    from numpy import std
    from numpy import append
    from numpy import array
    
    from sklearn.metrics import confusion_matrix
    from sklearn.base import clone
    
    if balanceMethod is 'SMOTE':
        from imblearn.over_sampling import SMOTE
#==============================================================================
#         from collections import Counter
#==============================================================================
        
    elif balanceMethod is 'undersample':
        from imblearn.under_sampling import RandomUnderSampler
    
    #==========================================================================
    
    # find a list of the (numerically coded) class labels, arranged in ascending order
    classLabels = sorted(unique(Y)) # "sorted" sorts in ascending order by default
    
    # find number of classes from list of true labels
    numClasses = size(classLabels) # automatically finds number of classes [we use this to define the size of the confusion matrix later]
    
    # find number of meerkats from list of meerkat numbers
    numMeerkats = size(unique(N)) # automatically finds number of meerkats for which data was collected [we shall use this to determine how many pairs of training and testing indices need to be generated]
    
    #==========================================================================
    
    # -----INITIALISATION OF VARIABLES THAT WILL BE STORED FOR EACH MEERKAT----
    
    # test indices for each meerkat
    testIndices = []
    # predicted labels for each meerkat
    predictedLabels = []
    # initialising the confusion matrix for each meerkat
    confMat = []
    # initialising the sensitivity for each meerkat (it will contain the one-V/S-all sensitivity for each class)
    sensitivity_eachMeerkat = [zeros([1,numClasses])[0] for i in range(numMeerkats)]
    # initialising the specificty for each meerkat (it will contain the one-V/S-all specificity for each class)
    specificity_eachMeerkat = [zeros([1,numClasses])[0] for i in range(numMeerkats)]
    # initialising the precision for each meerkat (it will contain the one-V/S-all precision for each class)
    precision_eachMeerkat = [zeros([1,numClasses])[0] for i in range(numMeerkats)]
    # initialising the accuracy for each meerkat (sum of diagonal elements of the confusion matrix divided by the sum of all elements of the confusion matrix)
    accuracy_eachMeerkat = zeros([numMeerkats,])
    # initialising a list that will store the fitted model that was computed for each meerkat (from which each model's parameters may later be accessed)
    trainedModel_staticDynamic_eachMeerkat = [] # model trained for static versus dynamic activity classification for each meerkat
    trainedModel_vigRest_eachMeerkat = [] # model trained for vigilance versus resting classification for each meerkat
    trainedModel_foragRun_eachMeerkat = [] # model trained for foraging versus running classification for each meerkat
    
    #==========================================================================
    
    # ----------TRAINING THE MODEL AND TESTING IT FOR EACH MEERKAT-------------
    
    count = 0 # counts the number of iterations of the loop below
    
    for i in unique(N):      # For each meerkat
        
        # calculating the training and testing indices when the i-th meerkat is left out of the training phase
        """ When the i-th meerkat is left out, the testing indices will be those 
        rows that correpond to N==i, and the training indices will be all the other
        rows (i.e. for which N!=i) """
        test_idx = array([idx for idx in range(len(N)) if N[idx]==i])
        train_idx = array([idx for idx in range(len(N)) if N[idx]!=i])
    
        #----------------------------------------------------------------------
        
        # DOING STATIC VERSUS DYNAMIC CLASSIFICATION
        
        M1 = clone(threeModels[0][1]) # creating a new model (for this iteration) for static versus dynamic classification. We use "clone" to ensure that the model for each iteration is "brand-new" (and doesn't re-use the parameters computed in the last iteration!)
        
        X_SD = X[:,featureIdx[0]] # only the features in "featureIdx[0] will be used for static VS dynamic classification
        
        # Python doesn't like feature matrices to have just one feature (one column, i.e.). It asks one to reshape the feature matrix to FeatureMatrix.reshape(-1,1)
        if size(featureIdx[0])==1:
            
            # DO CLASS-BALANCING IF NEEDED, AND TRAIN THE MODEL ON BALANCED TRAINING DATA
            if balanceMethod is 'none':
                # don't change the class distributions in this case
                # train the static versus dynamic model and store its parameters in one go
                trainedModel_staticDynamic_eachMeerkat.append(M1.fit(X_SD[train_idx].reshape(-1,1),Y_SD[train_idx]))
                
            elif balanceMethod is 'SMOTE':
                # oversample using the function "SMOTE" in imbalanced-learn so that all classes are balanced
                X_SD_resampled, Y_SD_resampled = SMOTE(kind='svm').fit_sample(X_SD[train_idx].reshape(-1,1),Y_SD[train_idx])
    #==============================================================================
    #             # check if data has been resampled
    #             print(sorted(Counter(Y_SD_resampled).items()))
    #==============================================================================
                # train the static versus dynamic model and store its parameters in one go
                trainedModel_staticDynamic_eachMeerkat.append(M1.fit(X_SD_resampled.reshape(-1,1),Y_SD_resampled))
                
            elif balanceMethod is 'undersample':
                # undersample majority classes to minority class using the function "RandomUnderSampler", so that all classes are balanced
                X_SD_resampled, Y_SD_resampled = RandomUnderSampler(random_state=0).fit_sample(X_SD[train_idx].reshape(-1,1),Y_SD[train_idx])
    #==============================================================================
    #             # check if data has been resampled
    #             print(sorted(Counter(Y_SD_resampled).items()))
    #==============================================================================
                # train the static versus dynamic model and store its parameters in one go
                trainedModel_staticDynamic_eachMeerkat.append(M1.fit(X_SD_resampled.reshape(-1,1),Y_SD_resampled))
                
            # END of class-balancing (if needed) and model-training
                
            # TESTING THE MODEL TRAINED JUST ABOVE
            # using the model trained (fit) above, do predictions on the test set (in other words, test the model that was trained in this iteration on the current test fold for static versus dynamic activity classification)
            Y_SD_predicted = M1.predict(X_SD[test_idx].reshape(-1,1)) # predicted static and dynamic labels
             
            #=================================================================
            
        else:
            
            # DO CLASS-BALANCING IF NEEDED, AND TRAIN THE MODEL ON BALANCED TRAINING DATA
            if balanceMethod is 'none':
                # don't change the class distributions in this case
                # train the static versus dynamic model and store its parameters in one go
                trainedModel_staticDynamic_eachMeerkat.append(M1.fit(X_SD[train_idx],Y_SD[train_idx]))
                
            elif balanceMethod is 'SMOTE':
                # oversample using the function "SMOTE" in imbalanced-learn so that all classes are balanced
                X_SD_resampled, Y_SD_resampled = SMOTE(kind='svm').fit_sample(X_SD[train_idx],Y_SD[train_idx])
    #==============================================================================
    #             # check if data has been resampled
    #             print(sorted(Counter(Y_SD_resampled).items()))
    #==============================================================================
                # train the static versus dynamic model and store its parameters in one go
                trainedModel_staticDynamic_eachMeerkat.append(M1.fit(X_SD_resampled,Y_SD_resampled))
                
            elif balanceMethod is 'undersample':
                # undersample majority classes to minority class using the function "RandomUnderSampler", so that all classes are balanced
                X_SD_resampled, Y_SD_resampled = RandomUnderSampler(random_state=0).fit_sample(X_SD[train_idx],Y_SD[train_idx])
    #==============================================================================
    #             # check if data has been resampled
    #             print(sorted(Counter(Y_SD_resampled).items()))
    #==============================================================================
                # train the static versus dynamic model and store its parameters in one go
                trainedModel_staticDynamic_eachMeerkat.append(M1.fit(X_SD_resampled,Y_SD_resampled))
                
                # END of class-balancing (if needed) and model-training
                
            # TESTING THE MODEL TRAINED JUST ABOVE
            # using the model trained (fit) above, do predictions on the test set (in other words, test the model that was trained in this iteration on the current test fold for static versus dynamic activity classification)
            Y_SD_predicted = M1.predict(X_SD[test_idx]) # predicted static and dynamic labels
             
            # END of static-dynamic model-training and prediction on test-set
             
            #==================================================================
            
        
        # Y_SD_true = Y_SD[test_idx] # true (from video-labels) static and dynamic labels
        
        #----------------------------------------------------------------------
        
        # DOING VIGILANCE VERSUS RESTING CLASSIFICATION
        
        M2 = clone(threeModels[1][1]) # creating a new model (for this iteration) for vigilance versus resting classification
        
        X_VRst = X[:,featureIdx[1]] # only the features in "featureIdx[1] will be used for vigilance VS resting classification
                
        # in training data, choose only those labels which correspond to vigilance and resting (which are both static activities)
        vigRest_trainIdx = [idx for idx in train_idx if Y_SD[idx]==1]
        
        # find test indices. The vig V/S rest model will be tested only on those instances which were classified as static by "M1"
        vigRest_testIdx = test_idx[[i1 for i1 in range(len(Y_SD_predicted)) if Y_SD_predicted[i1]==1]]
        
        if len(vigRest_testIdx)!=0:     # it's quite possible that M1 classified all the test instances as "dynamic", and so there are none left to be classified as vigilance or rest. So, to avoid throwing up an error in the "M2.predict" line, check for this here
        
            # Python doesn't like feature matrices to have just one feature (one column, i.e.). It asks one to reshape the feature matrix to FeatureMatrix.reshape(-1,1)
            if size(featureIdx[1])==1:

                # DO CLASS-BALANCING IF NEEDED, AND TRAIN THE MODEL ON BALANCED TRAINING DATA
                if balanceMethod is 'none':
                    # don't change the class distributions in this case
                    # train the vigilance versus resting model and store its parameters in one go
                    trainedModel_vigRest_eachMeerkat.append(M2.fit(X_VRst[vigRest_trainIdx].reshape(-1,1),Y[vigRest_trainIdx]))
                    
                elif balanceMethod is 'SMOTE':
                    # oversample using the function "SMOTE" in imbalanced-learn so that all classes are balanced
                    X_VRst_resampled, Y_VRst_resampled = SMOTE(kind='svm').fit_sample(X_VRst[vigRest_trainIdx].reshape(-1,1),Y[vigRest_trainIdx])
        #==============================================================================
        #             # check if data has been resampled
        #             print(sorted(Counter(Y_SD_resampled).items()))
        #==============================================================================
                    # train the vigilance versus resting model and store its parameters in one go
                    trainedModel_vigRest_eachMeerkat.append(M2.fit(X_VRst_resampled.reshape(-1,1),Y_VRst_resampled))
                    
                elif balanceMethod is 'undersample':
                    # undersample majority classes to minority class using the function "RandomUnderSampler", so that all classes are balanced
                    X_VRst_resampled, Y_VRst_resampled = RandomUnderSampler(random_state=0).fit_sample(X_VRst[vigRest_trainIdx].reshape(-1,1),Y[vigRest_trainIdx])
        #==============================================================================
        #             # check if data has been resampled
        #             print(sorted(Counter(Y_SD_resampled).items()))
        #==============================================================================
                    # train the vigilance versus resting model and store its parameters in one go
                    trainedModel_vigRest_eachMeerkat.append(M2.fit(X_VRst_resampled.reshape(-1,1),Y_VRst_resampled))
                    
                # END of class-balancing (if needed) and model-training
                
                # test the vigilance VS rest model trained in this fold    
                Y_VRst_predicted = M2.predict(X_VRst[vigRest_testIdx].reshape(-1,1))
                
                #==============================================================
                
            else:                

                # DO CLASS-BALANCING IF NEEDED, AND TRAIN THE MODEL ON BALANCED TRAINING DATA
                if balanceMethod is 'none':
                    # don't change the class distributions in this case
                    # train the vigilance versus resting model and store its parameters in one go
                    trainedModel_vigRest_eachMeerkat.append(M2.fit(X_VRst[vigRest_trainIdx],Y[vigRest_trainIdx]))
                    
                elif balanceMethod is 'SMOTE':
                    # oversample using the function "SMOTE" in imbalanced-learn so that all classes are balanced
                    X_VRst_resampled, Y_VRst_resampled = SMOTE(kind='svm').fit_sample(X_VRst[vigRest_trainIdx],Y[vigRest_trainIdx])
        #==============================================================================
        #             # check if data has been resampled
        #             print(sorted(Counter(Y_SD_resampled).items()))
        #==============================================================================
                    # train the vigilance versus resting model and store its parameters in one go
                    trainedModel_vigRest_eachMeerkat.append(M2.fit(X_VRst_resampled,Y_VRst_resampled))
                    
                elif balanceMethod is 'undersample':
                    # undersample majority classes to minority class using the function "RandomUnderSampler", so that all classes are balanced
                    X_VRst_resampled, Y_VRst_resampled = RandomUnderSampler(random_state=0).fit_sample(X_VRst[vigRest_trainIdx],Y[vigRest_trainIdx])
        #==============================================================================
        #             # check if data has been resampled
        #             print(sorted(Counter(Y_SD_resampled).items()))
        #==============================================================================
                    # train the vigilance versus resting model and store its parameters in one go
                    trainedModel_vigRest_eachMeerkat.append(M2.fit(X_VRst_resampled,Y_VRst_resampled))
                    
                # END of class-balancing (if needed) and model-training
                
                #==============================================================
                
                # test the vigilance VS rest model        
                Y_VRst_predicted = M2.predict(X_VRst[vigRest_testIdx])
                
            # END of vigilance-resting model-training and prediction on test-set
                
            #==================================================================
            
            # find the true labels for the data rows (in "X") for which predictions were made above
            Y_VRst_true = Y[vigRest_testIdx]
                
        else:
        
            Y_VRst_predicted = array([]) # empty array
            Y_VRst_true = array([]) # also an empty array
        
        
        #----------------------------------------------------------------------
        
        # DOING FORAGING VERSUS RUNNING CLASSIFICATION
        
        M3 = clone(threeModels[2][1]) # creating a new model (for this iteration) for foraging versus running classification
        
        X_FR = X[:,featureIdx[2]] # only the features in "featureIdx[2] will be used for foraging VS running classification
                
        # in training data, choose only those labels which correspond to foraging and resting (which are both dynamic activities)
        frgRun_trainIdx = [idx for idx in train_idx if Y_SD[idx]==2]
        
        # find test indices. The foraging VS running model will be tested only on those instances which were classified as dynamic by "M1"
        frgRun_testIdx = test_idx[[i1 for i1 in range(len(Y_SD_predicted)) if Y_SD_predicted[i1]==2]]
        
        if len(frgRun_testIdx)!=0:     # it's quite possible that M1 classified all the test instances as "static", and so there are none left to be classified as foraging or running. So, to avoid throwing up an error in the "M3.predict" line, check for this here

            # Python doesn't like feature matrices to have just one feature (one column, i.e.). It asks one to reshape the feature matrix to FeatureMatrix.reshape(-1,1)
            if size(featureIdx[2])==1:

                # DO CLASS-BALANCING IF NEEDED, AND TRAIN THE MODEL ON BALANCED TRAINING DATA
                if balanceMethod is 'none':
                    # don't change the class distributions in this case
                    # train the foraging versus running model and store its parameters in one go
                    trainedModel_foragRun_eachMeerkat.append(M3.fit(X_FR[frgRun_trainIdx].reshape(-1,1),Y[frgRun_trainIdx]))
                    
                elif balanceMethod is 'SMOTE':
                    # oversample using the function "SMOTE" in imbalanced-learn so that all classes are balanced
                    X_FR_resampled, Y_FR_resampled = SMOTE(kind='svm').fit_sample(X_FR[frgRun_trainIdx],Y[frgRun_trainIdx])
        #==============================================================================
        #             # check if data has been resampled
        #             print(sorted(Counter(Y_SD_resampled).items()))
        #==============================================================================
                    # train the foraging versus running model and store its parameters in one go
                    trainedModel_foragRun_eachMeerkat.append(M3.fit(X_FR_resampled.reshape(-1,1),Y_FR_resampled))
                    
                elif balanceMethod is 'undersample':
                    # undersample majority classes to minority class using the function "RandomUnderSampler", so that all classes are balanced
                    X_FR_resampled, Y_FR_resampled = RandomUnderSampler(random_state=0).fit_sample(X_FR[frgRun_trainIdx],Y[frgRun_trainIdx])
        #==============================================================================
        #             # check if data has been resampled
        #             print(sorted(Counter(Y_SD_resampled).items()))
        #==============================================================================
                    # train the foraging versus running model and store its parameters in one go
                    trainedModel_foragRun_eachMeerkat.append(M3.fit(X_FR_resampled.reshape(-1,1),Y_FR_resampled))
                    
                # END of class-balancing (if needed) and model-training
                
                # test the vigilance VS rest model trained in this fold    
                Y_FR_predicted = M3.predict(X_FR[frgRun_testIdx].reshape(-1,1))
                
                #==============================================================
                
            else:
                
                # DO CLASS-BALANCING IF NEEDED, AND TRAIN THE MODEL ON BALANCED TRAINING DATA
                if balanceMethod is 'none':
                    # don't change the class distributions in this case
                    # train the foraging versus running model and store its parameters in one go
                    trainedModel_foragRun_eachMeerkat.append(M3.fit(X_FR[frgRun_trainIdx],Y[frgRun_trainIdx]))
                    
                elif balanceMethod is 'SMOTE':
                    # oversample using the function "SMOTE" in imbalanced-learn so that all classes are balanced
                    X_FR_resampled, Y_FR_resampled = SMOTE(kind='svm').fit_sample(X_FR[frgRun_trainIdx],Y[frgRun_trainIdx])
        #==============================================================================
        #             # check if data has been resampled
        #             print(sorted(Counter(Y_SD_resampled).items()))
        #==============================================================================
                    # train the foraging versus running model and store its parameters in one go
                    trainedModel_foragRun_eachMeerkat.append(M3.fit(X_FR_resampled,Y_FR_resampled))
                    
                elif balanceMethod is 'undersample':
                    # undersample majority classes to minority class using the function "RandomUnderSampler", so that all classes are balanced
                    X_FR_resampled, Y_FR_resampled = RandomUnderSampler(random_state=0).fit_sample(X_FR[frgRun_trainIdx],Y[frgRun_trainIdx])
        #==============================================================================
        #             # check if data has been resampled
        #             print(sorted(Counter(Y_SD_resampled).items()))
        #==============================================================================
                    # train the foraging versus running model and store its parameters in one go
                    trainedModel_foragRun_eachMeerkat.append(M3.fit(X_FR_resampled,Y_FR_resampled))
                    
                # END of class-balancing (if needed) and model-training
                
                # test the vigilance VS rest model trained in this fold    
                Y_FR_predicted = M3.predict(X_FR[frgRun_testIdx])
                
                #==============================================================
                
            # END of foraging-running model-training and prediction on test-set
                
            #==================================================================
    
            # find the true labels for the data rows (in "X") for which predictions were made above
            Y_FR_true = Y[frgRun_testIdx]
            
        else:
            
            Y_FR_predicted = array([]) # empty array
            Y_FR_true = array([]) # also an empty array
        
        #----------------------------------------------------------------------
        
        """ Perhaps it is worth noting at this point that the union of 
        "vigRest_testIdx" and "frgRun_testIdx" is exactly equal to "test_idx".
        This is because "Y_SD_predicted" can only contain either 1's or 2's."""
        
        #----------------------------------------------------------------------
        
        
        # COMPUTE FINAL PREDICTED LABELS AND, FROM THEM, THE PERFORMANCE METRICS
        
        # merging the "VR" and "FR" predicted and true vectors to calculate the confusion matrix
        Y_predicted = append(Y_VRst_predicted, Y_FR_predicted)
        Y_true = append(Y_VRst_true, Y_FR_true)
        
        # computing the confusion matrix calculated for the test data in this iteration
        C = confusion_matrix(Y_true,Y_predicted,labels=classLabels) # "true" classes along the rows and "predicted" classes along the columns
        confMat.append(C)
        
        # saving the test indices for this meerkat
        testIndices.append(append(vigRest_testIdx, frgRun_testIdx)) # saving the test indices in the same order as they've been processed above (which was, in turn, determined by the predictions of the static VS dynamic classifier)
                
        # saving the predicted labels for the test data used in this meerkat
        predictedLabels.append(Y_predicted)
                
        # calculating performance metrics for each class for this meerkat's data
        sensitivity_eachMeerkat[count], specificity_eachMeerkat[count], precision_eachMeerkat[count], accuracy_eachMeerkat[count] = computePerformanceMetrics(C)
        
        count += 1
                
        # END of all meerkat-wise calculations
        
    #==========================================================================
        
    # storing all meerkat-wise variables in "meerkatWiseResults"
    meerkatWiseResults = initialiseStruct()
    meerkatWiseResults.testIndices = testIndices
    meerkatWiseResults.predictedLabels = predictedLabels
    meerkatWiseResults.confMat = confMat
    meerkatWiseResults.sensitivity = sensitivity_eachMeerkat
    meerkatWiseResults.specificity = specificity_eachMeerkat
    meerkatWiseResults.precision = precision_eachMeerkat
    meerkatWiseResults.accuracy = accuracy_eachMeerkat
    meerkatWiseResults.trainedModel_staticDynamic = trainedModel_staticDynamic_eachMeerkat
    meerkatWiseResults.trainedModel_vigRest = trainedModel_vigRest_eachMeerkat
    meerkatWiseResults.trainedModel_foragRun = trainedModel_foragRun_eachMeerkat
    
    #==========================================================================
    
    # ------CALCULATING AGGREGATED (across all meerkats) PERFORMANCE METRICS------
    
    # calculating the aggregated confusion matrix
    confMat_allMeerkats = zeros([numClasses,numClasses])
    for i in range(numMeerkats):
        confMat_allMeerkats += confMat[i]
        
    # calculating aggregated performance metrics from the aggregated confusion matrix
    sensitivity_allMeerkats, specificity_allMeerkats, precision_allMeerkats, accuracy_allMeerkats = computePerformanceMetrics(confMat_allMeerkats)
    
    # storing all aggregated performance metrics in "aggregatedResults"
    aggregatedResults = initialiseStruct()
    aggregatedResults.confMat = confMat_allMeerkats
    aggregatedResults.sensitivity = sensitivity_allMeerkats
    aggregatedResults.specificity = specificity_allMeerkats
    aggregatedResults.precision = precision_allMeerkats
    aggregatedResults.accuracy = accuracy_allMeerkats
    
    #==========================================================================
    
    # -----------------CALCULATING MEAN PERFORMANCE METRICS--------------------
    
    # calculating the mean confusion matrix (across all meerkats)
    confMat_mean = divide(aggregatedResults.confMat,numMeerkats)
    # calculating the standard deviation of each respective entry in the confusion matrix (across all meerkats)
    confMat_std = std(meerkatWiseResults.confMat,axis=0,ddof=1)
    """ "axis" means that the corresponding elements across all the matrices 
    (all the [1,2] elements in each matrix, say) are assembled in a vector, and
    the std is calculated for this vector. "ddof" specifies the denominator of
    the std expression, given as N-ddof"""
    
    # calculating the mean sensitivity (across all meerkats)
    sensitivity_mean = mean(meerkatWiseResults.sensitivity,axis=0)
    # calculating the standard deviation of sensitivity (across all meerkats)
    sensitivity_std = std(meerkatWiseResults.sensitivity,axis=0,ddof=1)
    
    # calculating the mean specificity (across all meerkats)
    specificity_mean = mean(meerkatWiseResults.specificity,axis=0)
    # calculating the standard deviation of specificity (across all meerkats)
    specificity_std = std(meerkatWiseResults.specificity,axis=0,ddof=1)

    # calculating the mean precision (across all meerkats)
    precision_mean = mean(meerkatWiseResults.precision,axis=0)
    # calculating the standard deviation of precision (across all meerkats)
    precision_std = std(meerkatWiseResults.precision,axis=0,ddof=1) 
    
    # calculating the mean accuracy (across all meerkats)
    accuracy_mean = mean(meerkatWiseResults.accuracy,axis=0)
    # calculating the standard deviation of accuracy (across all meerkats)
    accuracy_std = std(meerkatWiseResults.accuracy,axis=0,ddof=1) 
    
    
    meanResults = initialiseStruct()
    meanResults.confMat_mean = confMat_mean
    meanResults.confMat_std = confMat_std
    meanResults.sensitivity_mean = sensitivity_mean
    meanResults.sensitivity_std = sensitivity_std
    meanResults.specificity_mean = specificity_mean
    meanResults.specificity_std = specificity_std
    meanResults.precision_mean = precision_mean
    meanResults.precision_std = precision_std
    meanResults.accuracy_mean = accuracy_mean
    meanResults.accuracy_std = accuracy_std
    
    #==========================================================================
    
    # finally, return all the three types of results
    return aggregatedResults, meanResults, meerkatWiseResults


#%% FUNCTION TO CARRY OUT EASY-ENSEMBLE PRE-BALANCING OF CLASSES AND COMPUTING AND COMPILING RESULTS FOR A GIVEN LIST OF MODELS FOR A GIVEN FEATURE-SET

def prebalancedKfoldResults(X,Y,models,Nfolds=10,Nsubsets=10):
    
    """
    OUTPUT:
    "results":
        This is a list which has as many rows as the number of models that were
        tested (i.e. as many rows as those in "models"). Each row has three
        entries which are, in order:
            "name": the name of the model for which the results were generated
            in this row
            "aggSubset": the confusion matrix and 4 performance metrics for each 
            class computed over the aggregated confusion matrix. This matrix is
            aggregated across each fold across each subset. Thus, for the default
            values, this is a matrix aggregated across 10*10 different confusion
            matrices
            "meanSubset": the mean and standard deviation of each of the 4 performance
            metrics for each class
    
    INPUTS:
    "X": the feature matrix, with rows corresponding to observations, and 
    columns to features
    "Y": the list of labels for the observations in "X". Its length needs to 
    be exactly equal to the number of rows in "X"
    "models": a list of models to test on this feature-set and set of labels,
    and has two entries, its name and the model itself. For instance, one row
    of "models" can be: ('NaiveBayes', GaussianNB())
    "Nfolds": number of folds for which k-fold cross-validation will be done. It
    is set to 10 by default, but can be changed, but has to be at least 2
    "Nsubsets": number of random subsets generated by the function "EasyEnsemble".
    Set to 10 by default, and has to be at least 1
    
    
    Written:    21 September, 2017
                Pritish Chakravarty
                LMAM, EPFL
    """
    
    # import required models
    import time
    
    from imblearn.ensemble import EasyEnsemble
#==============================================================================
#     from collections import Counter
#==============================================================================
    
    from crossValidationFunctions import kFoldResults
    from crossValidationFunctions import computePerformanceMetrics
    from crossValidationFunctions import initialiseStruct
    
    from numpy import zeros
    from numpy import unique
    from numpy import mean
    from numpy import std
    from numpy import asarray
    
    numClasses = len(unique(Y))
    
    # Generating an ensemble of "n_subsets" randomly undersampled subsets of "X" with balanced classes
    ee = EasyEnsemble(random_state=0, n_subsets=Nsubsets)
    x_resampled, y_resampled = ee.fit_sample(X,Y)
    # the shape of x_resampled is: [n_subsets, no. of data points per subset, no. of features for each data point]
    
    # evaluate each model in turn, and store the results of each
    results = [] # this list will store the name of the model, the "twice-aggregated" confusion matrix, and the three performance metrics
    
    for name,model in models:
        
        start_time0 = time.time() # start time for the computations to be done for this model
        
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
            
            # finding the results of classification for this subset
            aggregated, meanFold, foldWise = kFoldResults(model,x,y,numFolds=Nfolds)
            
             # saving the performance metrics obtained from each fold in this subset
            allSensitivities.append(foldWise.sensitivity)
            allSpecificities.append(foldWise.specificity)
            allPrecisions.append(foldWise.precision)
            allAccuracies.append(foldWise.accuracy)
            
            # updating the overall confusion matrix
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
            sensitivity_mean[actvtNum] = mean(allSensitivities[:,:,actvtNum],axis=0)
            specificity_mean[actvtNum] = mean(allSpecificities[:,:,actvtNum],axis=0)
            precision_mean[actvtNum] = mean(allPrecisions[:,:,actvtNum],axis=0)
            
            
            # calculating the standard deviations of the performance metrics across all folds and all subsets
            sensitivity_std[actvtNum] = std(allSensitivities[:,:,actvtNum],axis=0,ddof=1)
            specificity_std[actvtNum] = std(allSpecificities[:,:,actvtNum],axis=0,ddof=1)
            precision_std[actvtNum] = std(allPrecisions[:,:,actvtNum],axis=0,ddof=1)
            
            # END of class--wise performance-metric calculation
        
        accuracy_mean = mean(allAccuracies,axis=0)
        accuracy_std = std(allAccuracies,axis=0,ddof=1)
        
        meanSubset.sensitivity_mean = sensitivity_mean
        meanSubset.sensitivity_std = sensitivity_std
        meanSubset.specificity_mean = specificity_mean
        meanSubset.specificity_std = specificity_std
        meanSubset.precision_mean = precision_mean
        meanSubset.precision_std = precision_std
        meanSubset.accuracy_mean = accuracy_mean
        meanSubset.accuracy_std = accuracy_std
        
        #----------------------------------------------------------
        
        # finally, appending the model information and the results for this iteration to "results"
        results.append([name, aggSubset, meanSubset]) 
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
        
        msg = "%s finished running" % (name)
        print(msg)
        
        msg = "Time taken: %f s" % (time.time() - start_time0)
        print(msg)
        
    return results

        
#%% FUNCTION TO CALCULATE PERFORMANCE METRICS FROM CONFUSION MATRIX

def computePerformanceMetrics(confMat):
    
    """ This function accepts a square confusion matrix as input and outputs
    three performance metrics - sensitivity, specificity and precision. It is
    assumed that "true labels" are arranged along rows and "predicted labels" 
    along columns. Each of the three performance metrics will have as many
    elements as the number of rows (or columns, since the confusion matrix is
    square), and gives these metrics for each class. Further, if the size of 
    the confusion matrix is greater than 2x2, these metrics will be calculated
    in a "one versus all" fashion. All values are reported as percentages and,
    for each performance metric, are arranged as row vectors.
                                    
    Written:
        16 Aug, 2017
        Pritish Chakravarty
        LMAM, EPFL
    """
    
    # IMPORTING THE REQUIRED LIBRARIES
    from numpy import zeros
    from numpy import shape
    
    numClasses,n_cols = shape(confMat) # using "numClasses" to signify the number of rows
    if numClasses != n_cols:
        raise ValueError('The input confusion matrix is not square')
    
    sensitivity = zeros([1,numClasses])[0] # class-wise sensitivity (one-V/S-all)
    specificity = zeros([1,numClasses])[0] # class-wise specificity (one-V/S-all)
    precision = zeros([1,numClasses])[0] # class-wise precision (one-V/S-all)
    
    for i in range(numClasses):
        
        # calculating sensitivity
        sensitivity[i] = round((confMat[i,i] / sum(confMat[i,:])) * 100, 1)
        # sensitivity = true positives / total real positives (expressed as percentage up to one decimal point)
        
        # calculating specificity
        TN = sum(sum(confMat)) - sum(confMat[:,i]) - sum(confMat[i,:]) + confMat[i,i] # number of true negatives. Formula corrected on 9 March, 2018. Earlier I had mistakenly used the formula: TN = trace(confMat) - confMat[i,i]
        N = sum(sum(confMat)) - sum(confMat[i,:]) # total number of negatives. Simplified the formula to this on 9 March, 2018. Earlier it was N = sum(sum(confMat[where(arange(numClasses)!=i)[0],:]))
        specificity[i] = round((TN / N) * 100,1)
        # specificity = true negatives / total number of negatives (expressed as percentage up to one decimal point)
        
        # calculating precision
        precision[i] = round((confMat[i,i] / sum(confMat[:,i])) * 100,1)
        
        # end of class-wise computations
    
    accuracy = round(sum(confMat[range(numClasses),range(numClasses)]) / sum(sum(confMat)) * 100, 1)
    # accuracy = sum of diagonal elements of the confusion matrix divided by the sum of all elements of the confusion matrix
        
    return sensitivity, specificity, precision, accuracy


#%% FUNCTIONS TO COMPUTE RESULTS 