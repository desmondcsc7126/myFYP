from random import Random
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

import pandas as pd
import numpy as np
import json

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
import xgboost
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

recall_avg = 'weighted'
precision_avg = 'weighted'
overfitThreshold = 20

def knnModel(data_dict, model_dict, final_model_dict, x):

    highest = 0

    knn_model_dict = model_dict['KNN']

    knn_performance_dict = {
       
    }

    score_rank = []

    xtrain = data_dict['train']['x']
    ytrain = data_dict['train']['y']

    xval = data_dict['validation']['x']
    yval = data_dict['validation']['y']

    xtest = data_dict['testing']['x']
    ytest = data_dict['testing']['y']

    id = 0
    goodModelCount = 0
    for w in knn_model_dict['Parameter List']['weights']:
        for n in knn_model_dict['Parameter List']['p']:

            my_dict = {
                'weights' : w,
                'p' : n,
                'val_score' : [],
                'train_score' : [],
                'flag' : 0
            }

            for k in knn_model_dict['Parameter List']['n_neighbors']:

                score_rank_dict = {
                    'val_score' : 0,
                    'weights' : 0,
                    'p' : 0,
                    'n_neighbors' :0
                }

                knn = KNeighborsClassifier(n_neighbors = k, weights = w, p = n)
                knn.fit(xtrain,ytrain)
                ypred_val = knn.predict(xval)

                # print(ypred_val)

                train_score = knn.score(xtrain,ytrain)
                val_score = accuracy_score(yval, ypred_val)

                my_dict['val_score'].append(val_score)
                my_dict['train_score'].append(train_score)

                # print('train: ',train_score,'   validate_score: ',val_score)

                goodModel = abs(train_score - val_score) / max(train_score, val_score) * 100 < overfitThreshold

                if goodModel:
                    score_rank_dict['val_score'] = val_score
                    score_rank_dict['weights'] = w
                    score_rank_dict['p'] = n
                    score_rank_dict['n_neighbors'] = k

                    score_rank.append(score_rank_dict.copy())

                if val_score > highest:
                    if goodModel:
                        knn_model_dict['Optimal Parameter']['weights'] = w
                        knn_model_dict['Optimal Parameter']['n_neighbors'] = k
                        knn_model_dict['Optimal Parameter']['p'] = n
                        goodModelCount += 1

                    elif goodModel == False and goodModelCount == 0:
                        knn_model_dict['Optimal Parameter']['weights'] = w
                        knn_model_dict['Optimal Parameter']['n_neighbors'] = k
                        knn_model_dict['Optimal Parameter']['p'] = n

                    highest = val_score
                    highest_train = train_score

            knn_performance_dict[id] = my_dict.copy()
            id += 1

    score_rank = sorted(score_rank, key=lambda x: x['val_score'], reverse = True)
    
    validModel = False
    i = 0
    # knn_final_model = knn_model_dict['Optimal Model']

    weights = knn_model_dict['Optimal Parameter']['weights']
    p = knn_model_dict['Optimal Parameter']['p']
    n_neighbors = knn_model_dict['Optimal Parameter']['n_neighbors']

    # knn_final_model = KNeighborsClassifier(weights=knn_model_dict['Optimal Parameter']['weights'], p=knn_model_dict['Optimal Parameter']['p'], n_neighbors=knn_model_dict['Optimal Parameter']['n_neighbors'] )
    knn_final_model = KNeighborsClassifier(weights=weights, p=p, n_neighbors=n_neighbors )
    xtrainFinal = data_dict['trainActual']['x']
    ytrainFinal = data_dict['trainActual']['y']
    scoreDictLength = len(score_rank)

    while validModel == False:

        knn_final_model.fit(xtrainFinal,ytrainFinal)
        ypred = knn_final_model.predict(xtest)

        print(highest)
        print(highest_train)

        test_score = accuracy_score(ytest,ypred)

        print(test_score)

        goodFinalModel = abs(test_score - highest) / max(test_score, highest) * 100 < overfitThreshold

        # print(precision_score(ytest,ypred))
        # print(recall_score(ytest,ypred))

        print(ypred)

        if goodFinalModel:

            classReport = classification_report(ytest, ypred, output_dict = True)

            knn_model_dict['Accuracy'] = classReport['accuracy']
            for label in knn_model_dict['label_dict']:
                knn_model_dict['label_dict'][label]['precision'] = round(classReport[str(label)]['precision'],2)
                knn_model_dict['label_dict'][label]['recall'] = round(classReport[str(label)]['recall'],2)
          
            # knn_model_dict['Optimal Model'] = json.dumps(knn_final_model)

            validModel = True

        else:

            i += 1
            if i >= scoreDictLength:
                break

            dict = score_rank[i]
            knn_final_model = KNeighborsClassifier(weights = dict['weights'], p = dict['p'], n_neighbors = dict['n_neighbors'])

            knn_model_dict['Optimal Parameter']['weights'] = dict['weights']
            knn_model_dict['Optimal Parameter']['n_neighbors'] = dict['n_neighbors']
            knn_model_dict['Optimal Parameter']['p'] = dict['p']

    if i >= scoreDictLength:
        knn_final_model = KNeighborsClassifier(weights=weights, p=p, n_neighbors=n_neighbors)
        knn_final_model.fit(xtrainFinal,ytrainFinal)
        ypred = knn_final_model.predict(xtest)

        classReport = classification_report(ytest, ypred, output_dict = True)
        knn_model_dict['Accuracy'] = classReport['accuracy']
        for label in knn_model_dict['label_dict']:
            knn_model_dict['label_dict'][label]['precision'] = round(classReport[str(label)]['precision'],2)
            knn_model_dict['label_dict'][label]['recall'] = round(classReport[str(label)]['recall'],2)

        knn_model_dict['Optimal Parameter']['weights'] = dict['weights']
        knn_model_dict['Optimal Parameter']['n_neighbors'] = dict['n_neighbors']
        knn_model_dict['Optimal Parameter']['p'] = dict['p']

    for key in knn_performance_dict:
        if knn_performance_dict[key]['weights'] == knn_model_dict['Optimal Parameter']['weights'] and knn_performance_dict[key]['p'] == knn_model_dict['Optimal Parameter']['p']:
            knn_performance_dict[key]['flag'] = 1
            break
    
    knn_model_dict['accuracy_record'] = knn_performance_dict.copy()
    final_model_dict['KNN'] = knn_model_dict.copy()

    x.updateProgress(40, 'K Nearest Neighbour modelling done')
    print('KNN')
    return True

def lgrModel(data_dict, model_dict, final_model_dict, x):

    highest = 0

    lgr_model_dict = model_dict['LGR']

    lgr_performance_dict = {
       
    }

    score_rank = []

    xtrain = data_dict['train']['x']
    ytrain = data_dict['train']['y']

    xval = data_dict['validation']['x']
    yval = data_dict['validation']['y']

    xtest = data_dict['testing']['x']
    ytest = data_dict['testing']['y']

    id = 0

    para_dict = lgr_model_dict['para_combination']
    goodModelCount = 0

    for key in para_dict:

        parameters = para_dict[key]
        solver = parameters['solver']
        penaltyList = parameters['penalty']

        for penalty in penaltyList:

            my_dict = {
                'solver' : solver,
                'penalty' : penalty,
                'val_score' : [],
                'train_score' : [],
                'flag' : 0
            }

            for c in lgr_model_dict['Parameter List']['C']:

                score_rank_dict = {
                    'val_score' : 0,
                    'solver' : 0,
                    'penalty' : 0,
                    'C' :0
                }

                lgr = LogisticRegression(solver=solver,penalty=penalty, C=c,max_iter=10000)

                lgr.fit(xtrain,ytrain)
                ypred_val = lgr.predict(xval)

                # print(ypred_val)

                train_score = lgr.score(xtrain,ytrain)
                val_score = accuracy_score(yval, ypred_val)

                my_dict['val_score'].append(val_score)
                my_dict['train_score'].append(train_score)

                # print('train: ',train_score,'   validate_score: ',val_score)

                goodModel = abs(train_score - val_score) / max(train_score, val_score) * 100 < overfitThreshold

                if goodModel:
                    score_rank_dict['val_score'] = val_score
                    score_rank_dict['solver'] = solver
                    score_rank_dict['penalty'] = penalty
                    score_rank_dict['C'] = c

                    score_rank.append(score_rank_dict.copy())

                if val_score > highest:
                    if goodModel:
                        lgr_model_dict['Optimal Parameter']['solver'] = solver
                        lgr_model_dict['Optimal Parameter']['penalty'] = penalty
                        lgr_model_dict['Optimal Parameter']['C'] = c
                        goodModelCount += 1
                    
                    elif goodModel == False and goodModelCount == 0:
                        lgr_model_dict['Optimal Parameter']['solver'] = solver
                        lgr_model_dict['Optimal Parameter']['penalty'] = penalty
                        lgr_model_dict['Optimal Parameter']['C'] = c
                    # lgr_model_dict['Optimal Model'] = lgr
                    highest = val_score
                    highest_train = train_score

            lgr_performance_dict[id] = my_dict.copy()
            id += 1

    score_rank = sorted(score_rank, key=lambda x: x['val_score'], reverse = True)
    
    validModel = False
    i = 0
    # knn_final_model = knn_model_dict['Optimal Model']
    solver = lgr_model_dict['Optimal Parameter']['solver']
    penalty = lgr_model_dict['Optimal Parameter']['penalty']
    C=lgr_model_dict['Optimal Parameter']['C']

    # lgr_final_model = LogisticRegression(solver=lgr_model_dict['Optimal Parameter']['solver'], penalty=lgr_model_dict['Optimal Parameter']['penalty'], C=lgr_model_dict['Optimal Parameter']['C'], max_iter=10000 )
    lgr_final_model = LogisticRegression(solver = solver, penalty = penalty, C = C, max_iter = 10000)
    xtrainFinal = data_dict['trainActual']['x']
    ytrainFinal = data_dict['trainActual']['y']

    scoreDictLength = len(score_rank)

    while validModel == False:

        lgr_final_model.fit(xtrainFinal,ytrainFinal)
        ypred = lgr_final_model.predict(xtest)

        print(highest)
        print(highest_train)

        test_score = accuracy_score(ytest,ypred)

        print(test_score)

        goodFinalModel = abs(test_score - highest) / max(test_score, highest) * 100 < overfitThreshold

        # print(precision_score(ytest,ypred))
        # print(recall_score(ytest,ypred))

        print(ypred)

        if goodFinalModel:

            classReport = classification_report(ytest, ypred, output_dict = True)

            lgr_model_dict['Accuracy'] = classReport['accuracy']
            for label in lgr_model_dict['label_dict']:
                lgr_model_dict['label_dict'][label]['precision'] = round(classReport[str(label)]['precision'],2)
                lgr_model_dict['label_dict'][label]['recall'] = round(classReport[str(label)]['recall'],2)
            # lgr_model_dict['Optimal Model'] = lgr_final_model

            validModel = True

        else:
            i += 1

            if i >= scoreDictLength:
                break

            dict = score_rank[i]
            lgr_final_model = LogisticRegression(solver = dict['solver'], penalty = dict['penalty'], C = dict['C'], max_iter = 10000)

            lgr_model_dict['Optimal Parameter']['solver'] = dict['solver']
            lgr_model_dict['Optimal Parameter']['penalty'] = dict['penalty']
            lgr_model_dict['Optimal Parameter']['C'] = dict['C']

    if i >= scoreDictLength:
        lgr_final_model = LogisticRegression(solver = solver, penalty = penalty, C = C, max_iter = 10000)
        lgr_final_model.fit(xtrainFinal,ytrainFinal)
        ypred = lgr_final_model.predict(xtest)

        classReport = classification_report(ytest, ypred, output_dict = True)
        lgr_model_dict['Accuracy'] = classReport['accuracy']
        for label in lgr_model_dict['label_dict']:
            lgr_model_dict['label_dict'][label]['precision'] = round(classReport[str(label)]['precision'],2)
            lgr_model_dict['label_dict'][label]['recall'] = round(classReport[str(label)]['recall'],2)

        lgr_model_dict['Optimal Parameter']['solver'] = solver
        lgr_model_dict['Optimal Parameter']['penalty'] = penalty
        lgr_model_dict['Optimal Parameter']['C'] = C

    for key in lgr_performance_dict:
        if lgr_performance_dict[key]['solver'] == lgr_model_dict['Optimal Parameter']['solver'] and lgr_performance_dict[key]['penalty'] == lgr_model_dict['Optimal Parameter']['penalty']:
            lgr_performance_dict[key]['flag'] = 1
            break
    
    lgr_model_dict['accuracy_record'] = lgr_performance_dict.copy()
    final_model_dict['LGR'] = lgr_model_dict.copy()

    x.updateProgress(40, 'Logistic Regression modelling done')
    print('LGR')
    return True

def rfcModel(data_dict, model_dict, final_model_dict, x):

    highest = 0

    rfc_model_dict = model_dict['RFC']

    rfc_performance_dict = {
       
    }

    score_rank = []

    xtrain = data_dict['train']['x']
    ytrain = data_dict['train']['y']

    xval = data_dict['validation']['x']
    yval = data_dict['validation']['y']

    xtest = data_dict['testing']['x']
    ytest = data_dict['testing']['y']

    id = 0
    goodModelCount = 0

    for criterion in rfc_model_dict['Parameter List']['criterion']:
        for min_split in rfc_model_dict['Parameter List']['min_samples_split']:

            for ccp_alpha in rfc_model_dict['Parameter List']['ccp_alpha']:
                my_dict = {
                    'criterion' : criterion,
                    'min_samples_split' : min_split,
                    'ccp_alpha' : ccp_alpha,
                    'val_score' : [],
                    'train_score' : [],
                    'flag' : 0
                }
                for max_depth in rfc_model_dict['Parameter List']['max_depth']:

                    score_rank_dict = {
                        'val_score' : 0,
                        'criterion' : criterion,
                        'min_samples_split' : min_split,
                        'max_depth' : 0,
                        'ccp_alpha' : 0
                    }

                    rfc = RandomForestClassifier(criterion = criterion, min_samples_split = min_split, max_depth = max_depth, ccp_alpha = ccp_alpha)

                    rfc.fit(xtrain,ytrain)
                    ypred_val = rfc.predict(xval)

                    # print(ypred_val)

                    train_score = rfc.score(xtrain,ytrain)
                    val_score = accuracy_score(yval, ypred_val)

                    my_dict['val_score'].append(val_score)
                    my_dict['train_score'].append(train_score)

                    # print('train: ',train_score,'   validate_score: ',val_score)

                    goodModel = abs(train_score - val_score) / max(train_score, val_score) * 100 < overfitThreshold

                    if goodModel:
                        score_rank_dict['val_score'] = val_score
                        score_rank_dict['min_samples_split'] = min_split
                        score_rank_dict['criterion'] = criterion
                        score_rank_dict['max_depth'] = max_depth
                        score_rank_dict['ccp_alpha'] = ccp_alpha

                        score_rank.append(score_rank_dict.copy())

                    if val_score > highest:
                        if goodModel:
                            rfc_model_dict['Optimal Parameter']['criterion'] = criterion
                            rfc_model_dict['Optimal Parameter']['min_samples_split'] = min_split
                            rfc_model_dict['Optimal Parameter']['max_depth'] = max_depth
                            rfc_model_dict['Optimal Parameter']['ccp_alpha'] = ccp_alpha
                            goodModelCount += 1

                        elif goodModel == False and goodModelCount == 0:
                            rfc_model_dict['Optimal Parameter']['criterion'] = criterion
                            rfc_model_dict['Optimal Parameter']['min_samples_split'] = min_split
                            rfc_model_dict['Optimal Parameter']['max_depth'] = max_depth
                            rfc_model_dict['Optimal Parameter']['ccp_alpha'] = ccp_alpha
                            
                        # rfc_model_dict['Optimal Model'] = rfc
                        highest = val_score
                        highest_train = train_score

                rfc_performance_dict[id] = my_dict.copy()
                id += 1

    validModel = False
    i = 0

    score_rank = sorted(score_rank, key=lambda x: x['val_score'], reverse = True)

    criterion = rfc_model_dict['Optimal Parameter']['criterion']
    min_samples_split = rfc_model_dict['Optimal Parameter']['min_samples_split']
    max_depth = rfc_model_dict['Optimal Parameter']['max_depth']
    ccp_alpha = rfc_model_dict['Optimal Parameter']['ccp_alpha']

    # rfc_final_model = RandomForestClassifier(criterion = rfc_model_dict['Optimal Parameter']['criterion'], min_samples_split=rfc_model_dict['Optimal Parameter']['min_samples_split'], max_depth=rfc_model_dict['Optimal Parameter']['max_depth'], ccp_alpha = rfc_model_dict['Optimal Parameter']['ccp_alpha'])
    rfc_final_model = RandomForestClassifier(criterion=criterion, min_samples_split=min_samples_split, max_depth=max_depth, ccp_alpha=ccp_alpha)
    xtrainFinal = data_dict['trainActual']['x']
    ytrainFinal = data_dict['trainActual']['y']

    scoreDictLength = len(score_rank)

    while validModel == False:

        rfc_final_model.fit(xtrainFinal,ytrainFinal)
        ypred = rfc_final_model.predict(xtest)

        print(highest)
        print(highest_train)

        test_score = accuracy_score(ytest,ypred)

        print(test_score)

        goodFinalModel = abs(test_score - highest) / max(test_score, highest) * 100 < overfitThreshold

        # print(precision_score(ytest,ypred))
        # print(recall_score(ytest,ypred))

        print(ypred)

        if goodFinalModel:

            classReport = classification_report(ytest, ypred, output_dict = True)

            rfc_model_dict['Accuracy'] = classReport['accuracy']
            for label in rfc_model_dict['label_dict']:
                rfc_model_dict['label_dict'][label]['precision'] = round(classReport[str(label)]['precision'],2)
                rfc_model_dict['label_dict'][label]['recall'] = round(classReport[str(label)]['recall'],2)
            # rfc_model_dict['Optimal Model'] = rfc_final_model

            validModel = True

        else:
            i += 1

            if i >= scoreDictLength:
                break

            dict = score_rank[i]
            rfc_final_model = RandomForestClassifier(criterion = dict['criterion'], min_samples_split = dict['min_samples_split'], max_depth = dict['max_depth'], ccp_alpha = dict['ccp_alpha'])

            rfc_model_dict['Optimal Parameter']['criterion'] = dict['criterion']
            rfc_model_dict['Optimal Parameter']['min_samples_split'] = dict['min_samples_split']
            rfc_model_dict['Optimal Parameter']['max_depth'] = dict['max_depth']
            rfc_model_dict['Optimal Parameter']['ccp_alpha'] = dict['ccp_alpha']

    # print(knn_performance_dict)
    # print(score_rank)

    if i >= scoreDictLength:
        rfc_final_model = RandomForestClassifier(criterion=criterion, min_samples_split=min_samples_split, max_depth=max_depth, ccp_alpha=ccp_alpha)
        rfc_final_model.fit(xtrainFinal,ytrainFinal)
        ypred = rfc_final_model.predict(xtest)

        classReport = classification_report(ytest, ypred, output_dict = True)
        rfc_model_dict['Accuracy'] = classReport['accuracy']
        for label in rfc_model_dict['label_dict']:
            rfc_model_dict['label_dict'][label]['precision'] = classReport[str(label)]['precision']
            rfc_model_dict['label_dict'][label]['recall'] = classReport[str(label)]['recall']
        
        rfc_model_dict['Optimal Parameter']['criterion'] = criterion
        rfc_model_dict['Optimal Parameter']['min_samples_split'] = min_samples_split
        rfc_model_dict['Optimal Parameter']['max_depth'] = max_depth
        rfc_model_dict['Optimal Parameter']['ccp_alpha'] = ccp_alpha

    for key in rfc_performance_dict:
        if rfc_performance_dict[key]['criterion'] == rfc_model_dict['Optimal Parameter']['criterion'] and rfc_performance_dict[key]['min_samples_split'] == rfc_model_dict['Optimal Parameter']['min_samples_split'] and rfc_performance_dict[key]['ccp_alpha'] == rfc_model_dict['Optimal Parameter']['ccp_alpha']:
            rfc_performance_dict[key]['flag'] = 1
            break
    
    rfc_model_dict['accuracy_record'] = rfc_performance_dict.copy()
    final_model_dict['RFC'] = rfc_model_dict.copy()
    x.updateProgress(40, 'Random Forest modelling done')
    
    print('RFC Done')
    return True

def dtModel(data_dict, model_dict, final_model_dict, x):

    highest = 0

    dt_model_dict = model_dict['DT']

    dt_performance_dict = {
       
    }

    score_rank = []

    xtrain = data_dict['train']['x']
    ytrain = data_dict['train']['y']

    xval = data_dict['validation']['x']
    yval = data_dict['validation']['y']

    xtest = data_dict['testing']['x']
    ytest = data_dict['testing']['y']

    id = 0
    goodModelCount = 0

    for criterion in dt_model_dict['Parameter List']['criterion']:
        for min_split in dt_model_dict['Parameter List']['min_samples_split']:

            for ccp_alpha in dt_model_dict['Parameter List']['ccp_alpha']:
                my_dict = {
                    'criterion' : criterion,
                    'min_samples_split' : min_split,
                    'ccp_alpha' : ccp_alpha,
                    'val_score' : [],
                    'train_score' : [],
                    'flag' : 0
                }
                for max_depth in dt_model_dict['Parameter List']['max_depth']:

                    score_rank_dict = {
                        'val_score' : 0,
                        'criterion' : criterion,
                        'min_samples_split' : min_split,
                        'max_depth' : 0,
                        'ccp_alpha' : 0
                    }

                    dt = DecisionTreeClassifier(criterion = criterion, min_samples_split = min_split, max_depth = max_depth, ccp_alpha = ccp_alpha)

                    dt.fit(xtrain,ytrain)
                    ypred_val = dt.predict(xval)

                    # print(ypred_val)

                    train_score = dt.score(xtrain,ytrain)
                    val_score = accuracy_score(yval, ypred_val)

                    my_dict['val_score'].append(val_score)
                    my_dict['train_score'].append(train_score)

                    # print('train: ',train_score,'   validate_score: ',val_score)

                    goodModel = abs(train_score - val_score) / max(train_score, val_score) * 100 < overfitThreshold

                    if goodModel:
                        score_rank_dict['val_score'] = val_score
                        score_rank_dict['min_samples_split'] = min_split
                        score_rank_dict['criterion'] = criterion
                        score_rank_dict['max_depth'] = max_depth
                        score_rank_dict['ccp_alpha'] = ccp_alpha

                        score_rank.append(score_rank_dict.copy())

                    if val_score > highest:
                        if goodModel:
                            dt_model_dict['Optimal Parameter']['criterion'] = criterion
                            dt_model_dict['Optimal Parameter']['min_samples_split'] = min_split
                            dt_model_dict['Optimal Parameter']['max_depth'] = max_depth
                            dt_model_dict['Optimal Parameter']['ccp_alpha'] = ccp_alpha
                            goodModelCount += 1

                        elif goodModel == False and goodModelCount == 0:
                            dt_model_dict['Optimal Parameter']['criterion'] = criterion
                            dt_model_dict['Optimal Parameter']['min_samples_split'] = min_split
                            dt_model_dict['Optimal Parameter']['max_depth'] = max_depth
                            dt_model_dict['Optimal Parameter']['ccp_alpha'] = ccp_alpha
                            
                        # rfc_model_dict['Optimal Model'] = rfc
                        highest = val_score
                        highest_train = train_score

                dt_performance_dict[id] = my_dict.copy()
                id += 1

    score_rank = sorted(score_rank, key=lambda x: x['val_score'], reverse = True)

    validModel = False
    i = 0

    criterion = dt_model_dict['Optimal Parameter']['criterion']
    min_samples_split = dt_model_dict['Optimal Parameter']['min_samples_split']
    max_depth = dt_model_dict['Optimal Parameter']['max_depth']
    ccp_alpha = dt_model_dict['Optimal Parameter']['ccp_alpha']

    # dt_final_model = DecisionTreeClassifier(criterion = dt_model_dict['Optimal Parameter']['criterion'], min_samples_split=dt_model_dict['Optimal Parameter']['min_samples_split'], max_depth=dt_model_dict['Optimal Parameter']['max_depth'], ccp_alpha = dt_model_dict['Optimal Parameter']['ccp_alpha'])
    dt_final_model = DecisionTreeClassifier(criterion=criterion, min_samples_split=min_samples_split, max_depth=max_depth, ccp_alpha=ccp_alpha)
    xtrainFinal = data_dict['trainActual']['x']
    ytrainFinal = data_dict['trainActual']['y']

    scoreDictLength = len(score_rank)

    while validModel == False:

        dt_final_model.fit(xtrainFinal,ytrainFinal)
        ypred = dt_final_model.predict(xtest)

        print(highest)
        print(highest_train)

        test_score = accuracy_score(ytest,ypred)

        print(test_score)

        goodFinalModel = abs(test_score - highest) / max(test_score, highest) * 100 < overfitThreshold

        # print(precision_score(ytest,ypred))
        # print(recall_score(ytest,ypred))

        print(ypred)

        if goodFinalModel:

            classReport = classification_report(ytest, ypred, output_dict = True)

            dt_model_dict['Accuracy'] = classReport['accuracy']
            for label in dt_model_dict['label_dict']:
                dt_model_dict['label_dict'][label]['precision'] = round(classReport[str(label)]['precision'],2)
                dt_model_dict['label_dict'][label]['recall'] = round(classReport[str(label)]['recall'],2)
            # rfc_model_dict['Optimal Model'] = rfc_final_model

            validModel = True

        else:
            i += 1

            if i >= scoreDictLength:
                break
            dict = score_rank[i]
            dt_final_model = DecisionTreeClassifier(criterion = dict['criterion'], min_samples_split = dict['min_samples_split'], max_depth = dict['max_depth'], ccp_alpha = dict['ccp_alpha'])

            dt_model_dict['Optimal Parameter']['criterion'] = dict['criterion']
            dt_model_dict['Optimal Parameter']['min_samples_split'] = dict['min_samples_split']
            dt_model_dict['Optimal Parameter']['max_depth'] = dict['max_depth']
            dt_model_dict['Optimal Parameter']['ccp_alpha'] = dict['ccp_alpha']

    # print(knn_performance_dict)
    # print(score_rank)

    if i >= scoreDictLength:
        dt_final_model = DecisionTreeClassifier(criterion=criterion, min_samples_split=min_samples_split, max_depth=max_depth, ccp_alpha=ccp_alpha)
        dt_final_model.fit(xtrainFinal,ytrainFinal)
        ypred = dt_final_model.predict(xtest)

        classReport = classification_report(ytest, ypred, output_dict = True)
        dt_model_dict['Accuracy'] = classReport['accuracy']
        for label in dt_model_dict['label_dict']:
            dt_model_dict['label_dict'][label]['precision'] = classReport[str(label)]['precision']
            dt_model_dict['label_dict'][label]['recall'] = classReport[str(label)]['recall']
        
        dt_model_dict['Optimal Parameter']['criterion'] = criterion
        dt_model_dict['Optimal Parameter']['min_samples_split'] = min_samples_split
        dt_model_dict['Optimal Parameter']['max_depth'] = max_depth
        dt_model_dict['Optimal Parameter']['ccp_alpha'] = ccp_alpha

    for key in dt_performance_dict:
        if dt_performance_dict[key]['criterion'] == dt_model_dict['Optimal Parameter']['criterion'] and dt_performance_dict[key]['min_samples_split'] == dt_model_dict['Optimal Parameter']['min_samples_split'] and dt_performance_dict[key]['ccp_alpha'] == dt_model_dict['Optimal Parameter']['ccp_alpha']:
            dt_performance_dict[key]['flag'] = 1
            break
    
    dt_model_dict['accuracy_record'] = dt_performance_dict.copy()
    final_model_dict['DT'] = dt_model_dict.copy()
    x.updateProgress(40, 'Decision Tree modelling done')
    print('Decision Tree Done')
    return True

def NBModel(data_dict, model_dict, final_model_dict, x):

    gnb = GaussianNB()
    mnb = MultinomialNB()
    bnb = BernoulliNB()

    final_model_dict['GNB'] = NBModelling(gnb, data_dict, model_dict['GNB']).copy()
    x.updateProgress(50, 'Gaussian Naive Bayes Modelling done')

    # final_model_dict['MNB'] = NBModelling(mnb, data_dict, model_dict['MNB']).copy()
    # x.updateProgress(55, 'Multinomial Naive Bayes Modelling done')

    final_model_dict['BNB'] = NBModelling(bnb, data_dict, model_dict['BNB']).copy()
    x.updateProgress(60, 'Bernoulli Naive Bayes Modelling done')
    print('NB done')
    return True

def NBModelling(model, data_dict, model_dict):

    xtrain = data_dict['train']['x']
    ytrain = data_dict['train']['y']

    xval = data_dict['validation']['x']
    yval = data_dict['validation']['y']

    xtest = data_dict['testing']['x']
    ytest = data_dict['testing']['y']

    model.fit(xtrain, ytrain)
    ypred_val = model.predict(xval)

    model_dict['accuracy_record']['train_score'] = model.score(xtrain,ytrain)
    model_dict['accuracy_record']['val_score'] = accuracy_score(yval, ypred_val)

    xtrainFinal = data_dict['trainActual']['x']
    ytrainFinal = data_dict['trainActual']['y']

    model.fit(xtrainFinal, ytrainFinal)
    ypred = model.predict(xtest)

    classReport = classification_report(ytest, ypred, output_dict = True)

    model_dict['Accuracy'] = classReport['accuracy']
    for label in model_dict['label_dict']:
        model_dict['label_dict'][label]['precision'] = round(classReport[str(label)]['precision'],2)
        model_dict['label_dict'][label]['recall'] = round(classReport[str(label)]['recall'],2)

    return model_dict

def XGBoostModel(data_dict, model_dict, final_model_dict, x):

    xtrain = data_dict['train']['x']
    ytrain = data_dict['train']['y']

    xval = data_dict['validation']['x']
    yval = data_dict['validation']['y']

    xtest = data_dict['testing']['x']
    ytest = data_dict['testing']['y']

    xtrainFinal = data_dict['trainActual']['x']
    ytrainFinal = data_dict['trainActual']['y']

    param = model_dict['XGBoost']['Parameter List']

    gridSearchXGB = GridSearchCV(estimator=xgboost.XGBClassifier(learning_rate = 0.1, early_stopping_rounds=10),
                                 param_grid=param,
                                cv = 2,
                                return_train_score=True,
                                verbose=False)

    gridSearchXGB.fit(xtrainFinal, ytrainFinal)

    df_cv_results = pd.DataFrame(gridSearchXGB.cv_results_)
    df_cv_results.to_csv('xgb.csv', index=False)

    df_cv_results = df_cv_results[['rank_test_score','mean_test_score','mean_train_score','param_n_estimators',
                                   'param_max_depth','param_subsample','param_colsample_bytree','param_gamma']]

    print(df_cv_results)

    return True

def XGBoostModel2(data_dict, model_dict, final_model_dict, x):

    xtrain = data_dict['train']['x']
    ytrain = data_dict['train']['y']

    xval = data_dict['validation']['x']
    yval = data_dict['validation']['y']

    xtest = data_dict['testing']['x']
    ytest = data_dict['testing']['y']

    xtrainFinal = data_dict['trainActual']['x']
    ytrainFinal = data_dict['trainActual']['y']

    highest = 0

    xgb_model_dict = model_dict['XGBoost']
    xgb_param = xgb_model_dict['Parameter List']

    xgb_performance_dict = {
       
    }

    score_rank = []

    eval_set = [(xval, yval)]

    id = 0
    goodModelCount = 0
    count = 0
    for n_estimators in xgb_param['n_estimators']:
        for subsample in xgb_param['subsample']:
            for colsample_bytree in xgb_param['colsample_bytree']:
                for gamma in xgb_param['gamma']:

                    my_dict = {
                        'n_estimators' : n_estimators,
                        'subsample' : subsample,
                        'colsample_bytree' : colsample_bytree,
                        'gamma' : gamma,
                        'val_score' : [],
                        'train_score' : [],
                        'flag' : 0
                    }

                    for max_depth in xgb_param['max_depth']:

                        score_rank_dict = {
                            'n_estimators' : n_estimators,
                            'subsample' : subsample,
                            'colsample_bytree' : colsample_bytree,
                            'gamma' : gamma,
                            'max_depth' : max_depth,
                            'val_score' : 0
                        }

                        xgb = xgboost.XGBClassifier(n_estimators = n_estimators,
                                                    subsample = subsample,
                                                    colsample_bytree = colsample_bytree,
                                                    gamma = gamma,
                                                    max_depth = max_depth,
                                                    learning_rate = 0.1,
                                                    eval_metric = 'auc')

                        xgb.fit(xtrain,ytrain,early_stopping_rounds=10,eval_set=eval_set, verbose = False)
                        ypred_val = xgb.predict(xval)

                        # print(ypred_val)

                        train_score = xgb.score(xtrain,ytrain)
                        val_score = accuracy_score(yval, ypred_val)

                        my_dict['val_score'].append(val_score)
                        my_dict['train_score'].append(train_score)

                        # print('train: ',train_score,'   validate_score: ',val_score)

                        goodModel = abs(train_score - val_score) / max(train_score, val_score) * 100 < overfitThreshold

                        if goodModel:
                            score_rank_dict['val_score'] = val_score
                            score_rank_dict['n_estimators'] = n_estimators
                            score_rank_dict['subsample'] = subsample
                            score_rank_dict['max_depth'] = max_depth
                            score_rank_dict['colsample_bytree'] = colsample_bytree
                            score_rank_dict['gamma'] = gamma

                            score_rank.append(score_rank_dict.copy())

                        if val_score > highest:
                            if goodModel:
                                xgb_model_dict['Optimal Parameter']['n_estimators'] = n_estimators
                                xgb_model_dict['Optimal Parameter']['subsample'] = subsample
                                xgb_model_dict['Optimal Parameter']['max_depth'] = max_depth
                                xgb_model_dict['Optimal Parameter']['colsample_bytree'] = colsample_bytree
                                xgb_model_dict['Optimal Parameter']['gamma'] = gamma
                                goodModelCount += 1

                            elif goodModel == False and goodModelCount == 0:
                                xgb_model_dict['Optimal Parameter']['n_estimators'] = n_estimators
                                xgb_model_dict['Optimal Parameter']['subsample'] = subsample
                                xgb_model_dict['Optimal Parameter']['max_depth'] = max_depth
                                xgb_model_dict['Optimal Parameter']['colsample_bytree'] = colsample_bytree
                                xgb_model_dict['Optimal Parameter']['gamma'] = gamma
                                
                            # rfc_model_dict['Optimal Model'] = rfc
                            highest = val_score
                            highest_train = train_score

                        count +=1
                        # print(count)

                    xgb_performance_dict[id] = my_dict.copy()
                    id += 1

    validModel = False
    i = 0

    score_rank = sorted(score_rank, key=lambda x: x['val_score'], reverse = True)

    n_estimators = xgb_model_dict['Optimal Parameter']['n_estimators']
    subsample = xgb_model_dict['Optimal Parameter']['subsample']
    colsample_bytree = xgb_model_dict['Optimal Parameter']['colsample_bytree']
    gamma = xgb_model_dict['Optimal Parameter']['gamma']
    max_depth = xgb_model_dict['Optimal Parameter']['max_depth']

    xgb_final_model = xgboost.XGBClassifier(n_estimators = n_estimators,
                                            subsample = subsample,
                                            colsample_bytree = colsample_bytree,
                                            gamma = gamma,
                                            max_depth = max_depth,
                                            learning_rate = 0.1)

    scoreDictLength = len(score_rank)

    while validModel == False:

        xgb_final_model.fit(xtrainFinal,ytrainFinal)
        ypred = xgb_final_model.predict(xtest)

        print(highest)
        print(highest_train)

        test_score = accuracy_score(ytest,ypred)

        print(test_score)

        goodFinalModel = abs(test_score - highest) / max(test_score, highest) * 100 < overfitThreshold

        # print(precision_score(ytest,ypred))
        # print(recall_score(ytest,ypred))

        print(ypred)

        if goodFinalModel:

            classReport = classification_report(ytest, ypred, output_dict = True)

            xgb_model_dict['Accuracy'] = classReport['accuracy']
            for label in xgb_model_dict['label_dict']:
                xgb_model_dict['label_dict'][label]['precision'] = round(classReport[str(label)]['precision'],2)
                xgb_model_dict['label_dict'][label]['recall'] = round(classReport[str(label)]['recall'],2)

            validModel = True

        else:
            i += 1
            if i >= score_rank_dict:
                break

            dict = score_rank[i]
            xgb_final_model = xgboost.XGBClassifier(n_estimators = dict['n_estimators'],
                                                    subsample = dict['subsample'],
                                                    colsample_bytree = dict['colsample_bytree'],
                                                    gamma = dict['gamma'],
                                                    max_depth = dict['max_depth'],
                                                    learning_rate = 0.1)
            

            xgb_model_dict['Optimal Parameter']['n_estimators'] = dict['n_estimators']
            xgb_model_dict['Optimal Parameter']['subsample'] = dict['subsample']
            xgb_model_dict['Optimal Parameter']['max_depth'] = dict['max_depth']
            xgb_model_dict['Optimal Parameter']['colsample_bytree'] = dict['colsample_bytree']
            xgb_model_dict['Optimal Parameter']['gamma'] = dict['gamma']

    if i >= scoreDictLength:
        xgb_final_model = xgboost.XGBClassifier(n_estimators = n_estimators,
                                                subsample = subsample,
                                                colsample_bytree = colsample_bytree,
                                                gamma = gamma,
                                                max_depth = max_depth,
                                                learning_rate = 0.1)

        xgb_final_model.fit(xtrainFinal,ytrainFinal)
        ypred = xgb_final_model.predict(xtest)

        classReport = classification_report(ytest, ypred, output_dict = True)
        xgb_model_dict['Accuracy'] = classReport['accuracy']
        for label in xgb_model_dict['label_dict']:
            xgb_model_dict['label_dict'][label]['precision'] = round(classReport[str(label)]['precision'],2)
            xgb_model_dict['label_dict'][label]['recall'] = round(classReport[str(label)]['recall'],2)

        xgb_model_dict['Optimal Parameter']['n_estimators'] = n_estimators
        xgb_model_dict['Optimal Parameter']['subsample'] = subsample
        xgb_model_dict['Optimal Parameter']['max_depth'] = max_depth
        xgb_model_dict['Optimal Parameter']['colsample_bytree'] = colsample_bytree
        xgb_model_dict['Optimal Parameter']['gamma'] = gamma

    for key in xgb_performance_dict:
        if (xgb_performance_dict[key]['n_estimators'] == xgb_model_dict['Optimal Parameter']['n_estimators'] and 
            xgb_performance_dict[key]['subsample'] == xgb_model_dict['Optimal Parameter']['subsample'] and
            xgb_performance_dict[key]['colsample_bytree'] == xgb_model_dict['Optimal Parameter']['colsample_bytree'] and
            xgb_performance_dict[key]['gamma'] == xgb_model_dict['Optimal Parameter']['gamma']):

            xgb_performance_dict[key]['flag'] = 1
            break
    
    xgb_model_dict['accuracy_record'] = xgb_performance_dict.copy()
    final_model_dict['XGBoost'] = xgb_model_dict.copy()
    x.updateProgress(40, 'XGBoost modelling done')
    print('XGB done')
    return True

def svmModel(data_dict, model_dict, final_model_dict, x):

    highest = 0

    svm_model_dict = model_dict['SVM']

    svm_performance_dict = {
       
    }

    count = 0

    score_rank = []

    xtrain = data_dict['train']['x']
    ytrain = data_dict['train']['y']

    xval = data_dict['validation']['x']
    yval = data_dict['validation']['y']

    xtest = data_dict['testing']['x']
    ytest = data_dict['testing']['y']

    id = 0
    goodModelCount = 0

    for kernel in svm_model_dict['Parameter List']['kernel']:
        for gamma in svm_model_dict['Parameter List']['gamma']:

            my_dict = {
                'kernel' : kernel,
                'gamma' : gamma,
                'val_score' : [],
                'train_score' : [],
                'flag' : 0
            }

            for C in svm_model_dict['Parameter List']['C']:

                score_rank_dict = {
                    'val_score' : 0,
                    'kernel' : 0,
                    'gamma' : 0,
                    'C' :0
                }

                svm = SVC(C = C, kernel = kernel, gamma = gamma)
                svm.fit(xtrain,ytrain)
                ypred_val = svm.predict(xval)

                # print(ypred_val)

                train_score = svm.score(xtrain,ytrain)
                val_score = accuracy_score(yval, ypred_val)

                my_dict['val_score'].append(val_score)
                my_dict['train_score'].append(train_score)

                # print('train: ',train_score,'   validate_score: ',val_score)

                goodModel = abs(train_score - val_score) / max(train_score, val_score) * 100 < overfitThreshold

                if goodModel:
                    score_rank_dict['val_score'] = val_score
                    score_rank_dict['kernel'] = kernel
                    score_rank_dict['gamma'] = gamma
                    score_rank_dict['C'] = C

                    score_rank.append(score_rank_dict.copy())

                if val_score > highest:
                    if goodModel:
                        svm_model_dict['Optimal Parameter']['kernel'] = kernel
                        svm_model_dict['Optimal Parameter']['C'] = C
                        svm_model_dict['Optimal Parameter']['gamma'] = gamma
                        goodModelCount += 1

                    elif goodModel == False and goodModelCount == 0:
                        svm_model_dict['Optimal Parameter']['kernel'] = kernel
                        svm_model_dict['Optimal Parameter']['C'] = C
                        svm_model_dict['Optimal Parameter']['gamma'] = gamma
                        
                    highest = val_score
                    highest_train = train_score
                count += 1
                print(count)
            svm_performance_dict[id] = my_dict.copy()
            id += 1

    score_rank = sorted(score_rank, key=lambda x: x['val_score'], reverse = True)
    
    validModel = False
    i = 0
    # knn_final_model = knn_model_dict['Optimal Model']

    C = svm_model_dict['Optimal Parameter']['C']
    kernel = svm_model_dict['Optimal Parameter']['kernel']
    gamma=svm_model_dict['Optimal Parameter']['gamma']

    # svm_final_model = SVC(C=svm_model_dict['Optimal Parameter']['C'], kernel=svm_model_dict['Optimal Parameter']['kernel'], gamma=svm_model_dict['Optimal Parameter']['gamma'] )
    svm_final_model  =SVC(C=C, kernel = kernel, gamma = gamma)
    xtrainFinal = data_dict['trainActual']['x']
    ytrainFinal = data_dict['trainActual']['y']

    scoreDictLength = len(score_rank)

    while validModel == False:

        svm_final_model.fit(xtrainFinal,ytrainFinal)
        ypred = svm_final_model.predict(xtest)

        print(highest)
        print(highest_train)

        test_score = accuracy_score(ytest,ypred)

        print(test_score)

        goodFinalModel = abs(test_score - highest) / max(test_score, highest) * 100 < overfitThreshold
        # print(precision_score(ytest,ypred))
        # print(recall_score(ytest,ypred))

        print(ypred)

        if goodFinalModel:

            classReport = classification_report(ytest, ypred, output_dict = True)

            svm_model_dict['Accuracy'] = classReport['accuracy']
            for label in svm_model_dict['label_dict']:
                svm_model_dict['label_dict'][label]['precision'] = round(classReport[str(label)]['precision'],2)
                svm_model_dict['label_dict'][label]['recall'] = round(classReport[str(label)]['recall'],2)
            # knn_model_dict['Optimal Model'] = json.dumps(knn_final_model)

            validModel = True

        else:
            i += 1
            if i >= scoreDictLength:
                break

            dict = score_rank[i]
            svm_final_model = SVC(C = dict['C'], kernel = dict['kernel'], gamma = dict['gamma'])

            svm_model_dict['Optimal Parameter']['kernel'] = dict['kernel']
            svm_model_dict['Optimal Parameter']['gamma'] = dict['gamma']
            svm_model_dict['Optimal Parameter']['C'] = dict['C']


    if i >= scoreDictLength:
        svm_final_model = SVC(kernel = kernel,gamma = gamma, C = C)
        svm_final_model.fit(xtrainFinal,ytrainFinal)
        ypred = svm_final_model.predict(xtest)

        classReport = classification_report(ytest, ypred, output_dict = True)
        svm_model_dict['Accuracy'] = classReport['accuracy']
        for label in svm_model_dict['label_dict']:
            svm_model_dict['label_dict'][label]['precision'] = round(classReport[str(label)]['precision'],2)
            svm_model_dict['label_dict'][label]['recall'] = round(classReport[str(label)]['recall'],2)

        svm_model_dict['Optimal Parameter']['kernel'] = kernel
        svm_model_dict['Optimal Parameter']['gamma'] = gamma
        svm_model_dict['Optimal Parameter']['C'] = C

    for key in svm_performance_dict:
        if svm_performance_dict[key]['kernel'] == svm_model_dict['Optimal Parameter']['kernel'] and svm_performance_dict[key]['gamma'] == svm_model_dict['Optimal Parameter']['gamma']:
            svm_performance_dict[key]['flag'] = 1
            break
    
    svm_model_dict['accuracy_record'] = svm_performance_dict.copy()
    final_model_dict['SVM'] = svm_model_dict.copy()

    x.updateProgress(40, 'Support Vector Machine modelling done')
    print('SVM done')
    return True

def dataModelling(df_modelling_dict, model_choice, x, modelType, label_class):

    target = df_modelling_dict['target']

    data_dict = {
        'categorical':[],
        'numerical':[],
        'train':{
            'x' : 0,
            'y' : 0
        },
        'validation':{
            'x' : 0,
            'y' : 0
        },
        'testing':{
            'x' : 0,
            'y' : 0
        },
        'trainActual':{
            'x' :0,
            'y' :0
        }
    }

    # ytrain, ytest, xtrain, xtest, xtrain_mmc, xtest_mmc, xtrain_std, xtest_std = dataSplitting(df_modelling_dict, target, data_dict)

    dataSplitting(df_modelling_dict, target, data_dict)

    # Start the modelling with each model

    model_dict = {
        'KNN' :{
            'Name':'K-Nearest Neighbour',
            'Parameter List':{
                'weights' : ['uniform','distance'],
                'n_neighbors' : [k for k in range(3,40,2)],
                'p' :[1,2]
            },
            'Optimal Parameter':{
                'weights' : None,
                'n_neighbors' : 3,
                'p' : 0
            },
            
            'Accuracy' : 0,
            'label_dict' : {

            },
            'main_axis' : {
                'value':[k for k in range(3,40,2)],
                'name':'n_neighbors'
                } 
        },

        'LGR' :{
            'Name':'Logistic Regression',
            'Parameter List' :{
                'C' : [0.01, 0.05, 0.1 ,0.5, 1.0, 5, 10],
                'penalty' : ['l1','l2','none'],
                'solver' : ['newton-cg', 'lbfgs', 'liblinear','sag','saga']
                
            },

            'para_combination' : {
                    0 : {'solver' : 'newton-cg', 'penalty' : ['l2','none']},
                    1 : {'solver' : 'lbfgs', 'penalty' : ['l2','none']},
                    2 : {'solver' : 'liblinear', 'penalty' : ['l1','l2']},
                    3 : {'solver' : 'sag', 'penalty' : ['l2','none']},
                    4 : {'solver' : 'saga', 'penalty' : ['l1','l2','none']}
            },

            'Optimal Parameter' : {
                'C' : 0,
                'penalty' : 0,
                'solver' : 0
            },

            
            'Accuracy' : 0,
            'label_dict' : {

            },
            'main_axis' : {
                'value':[0.01, 0.05, 0.1, 0.5, 1, 5, 10],
                'name' : 'C'
                }
        },

        'RFC' :{
            'Name':'Random Forest Classifier',
            'Parameter List' :{
                'max_depth' : [n for n in range(2,30,2)],
                'criterion' : ['gini', 'entropy', 'log_loss'],
                'min_samples_split' : [2,3,4],
                # 'ccp_alpha' : np.arange(0,1.1,0.1)
                'ccp_alpha' : [0, 0.01, 0.05, 0.1]
            },

            'Optimal Parameter' : {
                'max_depth' : 0,
                'criterion' : 0,
                'min_samples_split' : 0,
                'ccp_alpha' :0
            },

            
            'Accuracy': 0,
            'label_dict' : {

            },
            'main_axis' : {
                'value':[n for n in range(2,30,2)], 
                'name':'max_depth'
                }

        },

        'GNB' : {
            'Name':'Gaussian Naive Bayes',
            'Parameter List' : {},
            'Optimal Parameter' : {},
            'Accuracy' : 0,
            'label_dict' : {

            },
            'accuracy_record' : {
                'val_score' : 0,
                'train_score' : 0
            }
        },
        'MNB' : {
            'Name':'Multinomial Naive Bayes',
            'Parameter List' : {},
            'Optimal Parameter' : {},
            'Accuracy' : 0,
            'label_dict' : {

            },
            'accuracy_record' : {
                'val_score' : 0,
                'train_score' : 0
            }
        },
        'BNB' : {
            'Name':'Binomial Naive Bayes',
            'Parameter List' : {},
            'Optimal Parameter' : {},
            'Accuracy' : 0,
            'label_dict' : {

            },
            'accuracy_record' : {
                'val_score' : 0,
                'train_score' : 0
            }
        },

        'DT' :{
            'Name':'Decision Tree Classifier',
            'Parameter List' :{
                'max_depth' : [n for n in range(2,30,2)],
                'criterion' : ['gini', 'entropy', 'log_loss'],
                'min_samples_split' : [2,3,4],
                'ccp_alpha' : [0, 0.01, 0.05, 0.1]
            },

            'Optimal Parameter' : {
                'max_depth' : 0,
                'criterion' : 0,
                'min_samples_split' : 0,
                'ccp_alpha' :0
            },
           
            'Accuracy': 0,
            'label_dict' : {

            },
            'main_axis' : {
                'value':[n for n in range(2,30,2)], 
                'name':'max_depth'
                }
        },

        'XGBoost':{
            'Name':'XGBoost',
            'Parameter List':{
                # 'learning_rate' : 0.1,
                'n_estimators' : [1000,2000],
                'max_depth' : [n for n in range(3,21)],
                'subsample' : [0.8,1],
                'colsample_bytree' : [0.8,1],
                'gamma' : [0,1,5]
            },

            'Optimal Parameter':{
                'learning_rate' : 0.1,
                'n_estimators' : 0,
                'max_depth' : 0,
                'subsample' : 0,
                'colsample_bytree' : 0,
                'gamma' : 0,
                'eval_metric' : 'auc'
            },
            
            'Accuracy': 0,
            'label_dict' : {

            },
            'main_axis' : {
                'value':[n for n in range(3,21)],
                'name' : 'max_depth'
                }
        },

        'SVM' : {
            'Name':'Support Vector Machine Classifier',
            'Parameter List' : {
                'C' : [0.01, 0.1, 1],
                'kernel' : ['linear', 'rbf', 'sigmoid'],
                'gamma' : ['scale', 'auto']
            },

            'Optimal Parameter' : {
                'C' : 0,
                'kernel' : 'linear',
                'gamma' : 'auto'
            },

            'Accuracy' : 0,
            'label_dict' : {

            },
            'main_axis' : {
                'value':[0.01, 0.1, 1],
                'name' : 'C'
                }
        }
    }

    for modelName in model_dict:
        i = 0
        while i < len(label_class):

            label_dict = {
                'label' : label_class[i],
                'recall' : 0,
                'precision' : 0
            }
            model_dict[modelName]['label_dict'][i] = label_dict
            i += 1

    final_model_dict = {
        'modelType' : modelType,
        'model' : {}
    }

    func_dict = {
        'KNN':knnModel,
        'LGR':lgrModel,
        'RFC':rfcModel,
        'NB' :NBModel,
        'DT' : dtModel,
        'XGBoost' : XGBoostModel2,
        'SVM' : svmModel
    }

    for model in func_dict:
        if model in model_choice:
            try:
                func_dict[model](data_dict, model_dict, final_model_dict['model'], x)
            except:
                errorMsg = model + "Error"
                x.updateProgress(20,errorMsg)
                for key in final_model_dict['model']:
                    if key == model:
                        del final_model_dict['model'][model]


    for key in final_model_dict['model']:
        print('Model: ', key)
        print('Accuracy: ',final_model_dict['model'][key]['Accuracy'])
        print('Recall and Precision: ', final_model_dict['model'][key]['label_dict'])
        print('Optimal Parameter: ', final_model_dict['model'][key]['Optimal Parameter'])

    
    return final_model_dict

def dataSplitting(df_modelling_dict, target, col_dict):

    mmc = MinMaxScaler()
    std = StandardScaler()

    df_num = pd.DataFrame()
    df_cat = pd.DataFrame()

    if len(df_modelling_dict['numerical']) > 0:
        df_num = df_modelling_dict['numerical'].copy()

        for col_name in df_num:
            if col_name != target:
                col_dict['numerical'].append(col_name)

    if len(df_modelling_dict['categorical']) > 0:
        df_cat = df_modelling_dict['categorical'].copy()

        for col_name in df_cat:
            col_dict['categorical'].append(col_name)

    if len(df_num) > 0 and len(df_cat) > 0:
        df_modelling = pd.concat([df_num, df_cat],axis=1)

    elif len(df_num) > 0 and len(df_cat) == 0:
        df_modelling = df_num
    
    else:
        df_modelling = df_cat

    # Train test split

    y = df_modelling[target]
    x = df_modelling.drop([target],axis = 1)
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2)

    col_dict['testing']['y'] = ytest

    if len(df_num) > 0:
    
        # xtrain_mmc = xtrain.copy()
        # xtest_mmc = xtest.copy()
        xtrain_std = xtrain.copy()
        xtest_std = xtest.copy()

        # xtrain_mmc.loc[:,col_dict['numerical']] = mmc.fit_transform(xtrain_mmc.loc[:,col_dict['numerical']])
        # xtest_mmc.loc[:,col_dict['numerical']] = mmc.transform(xtest_mmc.loc[:,col_dict['numerical']])

        xtrain_std.loc[:,col_dict['numerical']] = std.fit_transform(xtrain_std.loc[:,col_dict['numerical']])
        xtest_std.loc[:,col_dict['numerical']] = std.transform(xtest_std.loc[:,col_dict['numerical']])

    xtr, xval, ytr, yval = train_test_split(xtrain_std, ytrain, test_size=0.2)

    col_dict['train']['x'] = xtr
    col_dict['train']['y'] = ytr

    col_dict['validation']['x'] = xval
    col_dict['validation']['y'] = yval

    col_dict['trainActual']['x'] = xtrain_std
    col_dict['trainActual']['y'] = ytrain

    col_dict['testing']['x'] = xtest_std
    
    return True

