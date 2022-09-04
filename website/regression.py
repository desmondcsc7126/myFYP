import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

from .dataModelling import dataSplitting

from sklearn.metrics import r2_score,mean_squared_error

overfitThreshold = 20

def linearRegressionModel(data_dict, model_dict, final_model_dict, x):

    highest = 0

    lr_model_dict = model_dict['LR']

    lr_performance_dict = {
       
    }

    score_rank = []

    xtrain = data_dict['train']['x']
    ytrain = data_dict['train']['y']

    xval = data_dict['validation']['x']
    yval = data_dict['validation']['y']

    xtest = data_dict['testing']['x']
    ytest = data_dict['testing']['y']

    xtrainFinal = data_dict['trainActual']['x']
    ytrainFinal = data_dict['trainActual']['y']

    id = 0
    goodModelCount = 0

    for fit_intercept in lr_model_dict['Parameter List']['fit_intercept']:
        for normalize in lr_model_dict['Parameter List']['normalize']:

            my_dict = {
                'fit_intercept' : str(fit_intercept),
                'normalize' : str(normalize),
                'val_score' : [],
                'train_score' : [],
                'flag' : 0
            }

            score_rank_dict = {
                'fit_intercept' : 0,
                'normalize' : 0,
                'val_score' : 0,
            }

            lr = LinearRegression(fit_intercept=fit_intercept, normalize=normalize)
            lr.fit(xtrain, ytrain)
            ypred_val = lr.predict(xval)

            train_score = lr.score(xtrain,ytrain)
            val_score = r2_score(yval, ypred_val)

            my_dict['val_score'].append(val_score)
            my_dict['train_score'].append(train_score)

            goodModel = abs(train_score - val_score) / max(train_score, val_score) * 100 < overfitThreshold

            if goodModel:
                score_rank_dict['val_score'] = val_score
                score_rank_dict['fit_intercept'] = fit_intercept
                score_rank_dict['normalize'] = normalize
                score_rank.append(score_rank_dict.copy())

            if val_score > highest:
                if goodModel:
                    lr_model_dict['Optimal Parameter']['fit_intercept'] = fit_intercept
                    lr_model_dict['Optimal Parameter']['normalize'] = normalize
                    goodModelCount += 1

                elif goodModel == False and goodModelCount == 0:
                    lr_model_dict['Optimal Parameter']['fit_intercept'] = fit_intercept
                    lr_model_dict['Optimal Parameter']['normalize'] = normalize

                highest = val_score

            lr_performance_dict[id] = my_dict.copy()
            id += 1

    score_rank = sorted(score_rank, key=lambda x: x['val_score'], reverse = True)
    print(score_rank)

    validModel = False
    i = 0

    fit_intercept = lr_model_dict['Optimal Parameter']['fit_intercept']
    normalize = lr_model_dict['Optimal Parameter']['normalize']
    # lr_final_model = LinearRegression(fit_intercept = lr_model_dict['Optimal Parameter']['fit_intercept'], normalize = lr_model_dict['Optimal Parameter']['normalize'])
    lr_final_model = LinearRegression(fit_intercept=fit_intercept, normalize=normalize)

    scoreDictLength = len(score_rank)

    while validModel == False:
        lr_final_model.fit(xtrainFinal, ytrainFinal)
        ypred = lr_final_model.predict(xtest)

        test_score = r2_score(ytest,ypred)
        mse = mean_squared_error(ytest,ypred)

        goodFinalModel = abs(test_score - highest) / max(test_score, highest) * 100 < overfitThreshold
        if goodFinalModel:
            lr_model_dict['r2_score'] = round(test_score,2)
            lr_model_dict['Mean Squared Error'] = round(mse,2)
            validModel = True

        else:
            i += 1
            if i >= scoreDictLength:
                lr_final_model = LinearRegression(fit_intercept=fit_intercept, normalize=normalize)
                lr_final_model.fit(xtrainFinal, ytrainFinal)
                ypred = lr_final_model.predict(xtest)
                test_score = r2_score(ytest,ypred)
                mse = mean_squared_error(ytest,ypred)
                lr_model_dict['r2_score'] = round(test_score,2)
                lr_model_dict['Mean Squared Error'] = round(mse,2)

                lr_model_dict['Optimal Parameter']['fit_intercept'] = fit_intercept
                lr_model_dict['Optimal Parameter']['normalize'] = normalize
                break

            dict = score_rank[i]
  
            lr_final_model = LinearRegression(fit_intercept = dict['fit_intercept'], normalize = dict['normalize'])
            lr_model_dict['Optimal Parameter']['fit_intercept'] = dict['fit_intercept']
            lr_model_dict['Optimal Parameter']['normalize'] = dict['normalize']

    for key in lr_performance_dict:
        if (str(lr_performance_dict[key]['fit_intercept']) == str(lr_model_dict['Optimal Parameter']['fit_intercept']) and
            str(lr_performance_dict[key]['normalize']) == str(lr_model_dict['Optimal Parameter']['normalize'])):
            lr_performance_dict[key]['flag'] = 1
            break

    lr_model_dict['accuracy_record'] = lr_performance_dict.copy()
    lr_model_dict['optimal model'] = lr_final_model
    final_model_dict['LR'] = lr_model_dict.copy()

    x.updateProgress(40,'Linear Regression Modelling Done')
    print('Linear Regression')

    return True

def DTRegressionModel(data_dict, model_dict, final_model_dict, x):

    highest = 0

    dtr_model_dict = model_dict['DTR']

    dtr_performance_dict = {
       
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

    for criterion in dtr_model_dict['Parameter List']['criterion']:
        for min_split in dtr_model_dict['Parameter List']['min_samples_split']:

            for ccp_alpha in dtr_model_dict['Parameter List']['ccp_alpha']:
                my_dict = {
                    'criterion' : criterion,
                    'min_samples_split' : min_split,
                    'ccp_alpha' : ccp_alpha,
                    'val_score' : [],
                    'train_score' : [],
                    'flag' : 0
                }
                for max_depth in dtr_model_dict['Parameter List']['max_depth']:

                    score_rank_dict = {
                        'val_score' : 0,
                        'criterion' : criterion,
                        'min_samples_split' : min_split,
                        'max_depth' : 0,
                        'ccp_alpha' : 0
                    }

                    dtr = DecisionTreeRegressor(criterion = criterion, min_samples_split = min_split, max_depth = max_depth, ccp_alpha = ccp_alpha)

                    dtr.fit(xtrain,ytrain)
                    ypred_val = dtr.predict(xval)

                    # print(ypred_val)

                    train_score = dtr.score(xtrain,ytrain)
                    val_score = r2_score(yval, ypred_val)

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
                            dtr_model_dict['Optimal Parameter']['criterion'] = criterion
                            dtr_model_dict['Optimal Parameter']['min_samples_split'] = min_split
                            dtr_model_dict['Optimal Parameter']['max_depth'] = max_depth
                            dtr_model_dict['Optimal Parameter']['ccp_alpha'] = ccp_alpha
                            goodModelCount += 1

                        elif goodModel == False and goodModelCount == 0:
                            dtr_model_dict['Optimal Parameter']['criterion'] = criterion
                            dtr_model_dict['Optimal Parameter']['min_samples_split'] = min_split
                            dtr_model_dict['Optimal Parameter']['max_depth'] = max_depth
                            dtr_model_dict['Optimal Parameter']['ccp_alpha'] = ccp_alpha
                            
                        # rfc_model_dict['Optimal Model'] = rfc
                        highest = val_score
                        highest_train = train_score

                dtr_performance_dict[id] = my_dict.copy()
                id += 1

    validModel = False
    i = 0

    score_rank = sorted(score_rank, key=lambda x: x['val_score'], reverse = True)

    criterion = dtr_model_dict['Optimal Parameter']['criterion']
    min_samples_split = dtr_model_dict['Optimal Parameter']['min_samples_split']
    max_depth = dtr_model_dict['Optimal Parameter']['max_depth']
    ccp_alpha = dtr_model_dict['Optimal Parameter']['ccp_alpha']

    # dtr_final_model = DecisionTreeRegressor(criterion = dtr_model_dict['Optimal Parameter']['criterion'], min_samples_split=dtr_model_dict['Optimal Parameter']['min_samples_split'], max_depth=dtr_model_dict['Optimal Parameter']['max_depth'], ccp_alpha = dtr_model_dict['Optimal Parameter']['ccp_alpha'])
    dtr_final_model = DecisionTreeRegressor(criterion=criterion,min_samples_split=min_samples_split, max_depth=max_depth, ccp_alpha=ccp_alpha)
    xtrainFinal = data_dict['trainActual']['x']
    ytrainFinal = data_dict['trainActual']['y']

    scoreDictLength = len(score_rank)

    while validModel == False:

        dtr_final_model.fit(xtrainFinal,ytrainFinal)
        ypred = dtr_final_model.predict(xtest)

        test_score = r2_score(ytest,ypred)
        mse = mean_squared_error(ytest, ypred)

        print(highest)
        print(test_score)

        goodFinalModel = abs(test_score - highest) / max(test_score, highest) * 100 < overfitThreshold

        # print(precision_score(ytest,ypred))
        # print(recall_score(ytest,ypred))



        if goodFinalModel:

            dtr_model_dict['r2_score'] = round(test_score,2)
            dtr_model_dict['Mean Squared Error'] = round(mse,2)
            # rfc_model_dict['Optimal Model'] = rfc_final_model

            validModel = True

        else:
            i += 1

            if i >= scoreDictLength:
                dtr_final_model = DecisionTreeRegressor(criterion=criterion,min_samples_split=min_samples_split, max_depth=max_depth, ccp_alpha=ccp_alpha)
                dtr_final_model.fit(xtrainFinal,ytrainFinal)
                ypred = dtr_final_model.predict(xtest)
                test_score = r2_score(ytest,ypred)
                mse = mean_squared_error(ytest, ypred)
                dtr_model_dict['r2_score'] = round(test_score,2)
                dtr_model_dict['Mean Squared Error'] = round(mse,2)

                dtr_model_dict['Optimal Parameter']['criterion'] = criterion
                dtr_model_dict['Optimal Parameter']['min_samples_split'] = min_samples_split
                dtr_model_dict['Optimal Parameter']['max_depth'] = max_depth
                dtr_model_dict['Optimal Parameter']['ccp_alpha'] = ccp_alpha
                break

            dict = score_rank[i]
            dtr_final_model = DecisionTreeRegressor(criterion = dict['criterion'], min_samples_split = dict['min_samples_split'], max_depth = dict['max_depth'], ccp_alpha = dict['ccp_alpha'])

            dtr_model_dict['Optimal Parameter']['criterion'] = dict['criterion']
            dtr_model_dict['Optimal Parameter']['min_samples_split'] = dict['min_samples_split']
            dtr_model_dict['Optimal Parameter']['max_depth'] = dict['max_depth']
            dtr_model_dict['Optimal Parameter']['ccp_alpha'] = dict['ccp_alpha']

    # print(knn_performance_dict)
    # print(score_rank)

    for key in dtr_performance_dict:
        if dtr_performance_dict[key]['criterion'] == dtr_model_dict['Optimal Parameter']['criterion'] and dtr_performance_dict[key]['min_samples_split'] == dtr_model_dict['Optimal Parameter']['min_samples_split'] and dtr_performance_dict[key]['ccp_alpha'] == dtr_model_dict['Optimal Parameter']['ccp_alpha']:
            dtr_performance_dict[key]['flag'] = 1
            break
    
    dtr_model_dict['accuracy_record'] = dtr_performance_dict.copy()
    dtr_model_dict['optimal model'] = dtr_final_model
    final_model_dict['DTR'] = dtr_model_dict.copy()

    x.updateProgress(40,'Decision Tree Regression Modelling Done')
    return True

def RFRegressionModel(data_dict, model_dict, final_model_dict, x):

    highest = 0

    rfr_model_dict = model_dict['RFR']

    rfr_performance_dict = {
       
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

    for criterion in rfr_model_dict['Parameter List']['criterion']:
        for min_split in rfr_model_dict['Parameter List']['min_samples_split']:

            for ccp_alpha in rfr_model_dict['Parameter List']['ccp_alpha']:
                my_dict = {
                    'criterion' : criterion,
                    'min_samples_split' : min_split,
                    'ccp_alpha' : ccp_alpha,
                    'val_score' : [],
                    'train_score' : [],
                    'flag' : 0
                }
                for max_depth in rfr_model_dict['Parameter List']['max_depth']:

                    score_rank_dict = {
                        'val_score' : 0,
                        'criterion' : criterion,
                        'min_samples_split' : min_split,
                        'max_depth' : 0,
                        'ccp_alpha' : 0
                    }

                    rfr = RandomForestRegressor(criterion = criterion, min_samples_split = min_split, max_depth = max_depth, ccp_alpha = ccp_alpha)

                    rfr.fit(xtrain,ytrain)
                    ypred_val = rfr.predict(xval)

                    # print(ypred_val)

                    train_score = rfr.score(xtrain,ytrain)
                    val_score = r2_score(yval, ypred_val)

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
                            rfr_model_dict['Optimal Parameter']['criterion'] = criterion
                            rfr_model_dict['Optimal Parameter']['min_samples_split'] = min_split
                            rfr_model_dict['Optimal Parameter']['max_depth'] = max_depth
                            rfr_model_dict['Optimal Parameter']['ccp_alpha'] = ccp_alpha
                            goodModelCount += 1

                        elif goodModel == False and goodModelCount == 0:
                            rfr_model_dict['Optimal Parameter']['criterion'] = criterion
                            rfr_model_dict['Optimal Parameter']['min_samples_split'] = min_split
                            rfr_model_dict['Optimal Parameter']['max_depth'] = max_depth
                            rfr_model_dict['Optimal Parameter']['ccp_alpha'] = ccp_alpha
                            
                        # rfc_model_dict['Optimal Model'] = rfc
                        highest = val_score
                        highest_train = train_score

                rfr_performance_dict[id] = my_dict.copy()
                id += 1

    validModel = False
    i = 0

    score_rank = sorted(score_rank, key=lambda x: x['val_score'], reverse = True)

    criterion = rfr_model_dict['Optimal Parameter']['criterion']
    min_samples_split = rfr_model_dict['Optimal Parameter']['min_samples_split']
    max_depth = rfr_model_dict['Optimal Parameter']['max_depth']
    ccp_alpha = rfr_model_dict['Optimal Parameter']['ccp_alpha']

    # rfc_final_model = RandomForestRegressor(criterion = rfr_model_dict['Optimal Parameter']['criterion'], min_samples_split=rfr_model_dict['Optimal Parameter']['min_samples_split'], max_depth=rfr_model_dict['Optimal Parameter']['max_depth'], ccp_alpha = rfr_model_dict['Optimal Parameter']['ccp_alpha'])
    rfc_final_model = RandomForestRegressor(criterion = criterion, min_samples_split=min_samples_split, max_depth = max_depth, ccp_alpha=ccp_alpha)
    xtrainFinal = data_dict['trainActual']['x']
    ytrainFinal = data_dict['trainActual']['y']

    scoreDictLength = len(score_rank)

    while validModel == False:

        rfc_final_model.fit(xtrainFinal,ytrainFinal)
        ypred = rfc_final_model.predict(xtest)

        test_score = r2_score(ytest,ypred)
        mse = mean_squared_error(ytest, ypred)

        print("Highest: ", highest)
        print("Test_score", test_score)

        goodFinalModel = abs(test_score - highest) / max(test_score, highest) * 100 < overfitThreshold

        # print(precision_score(ytest,ypred))
        # print(recall_score(ytest,ypred))

        # print(ypred)

        if goodFinalModel:

            rfr_model_dict['r2_score'] = round(test_score,2)
            rfr_model_dict['Mean Squared Error'] = round(mse,2)
            # rfc_model_dict['Optimal Model'] = rfc_final_model

            validModel = True

        else:
            i += 1

            if i >= scoreDictLength:

                rfc_final_model = RandomForestRegressor(criterion = criterion, min_samples_split=min_samples_split, max_depth = max_depth, ccp_alpha=ccp_alpha)
                rfc_final_model.fit(xtrainFinal,ytrainFinal)
                ypred = rfc_final_model.predict(xtest)
                test_score = r2_score(ytest,ypred)
                mse = mean_squared_error(ytest, ypred)
                rfr_model_dict['r2_score'] = round(test_score,2)
                rfr_model_dict['Mean Squared Error'] = round(mse,2)

                rfr_model_dict['Optimal Parameter']['criterion'] = criterion
                rfr_model_dict['Optimal Parameter']['min_samples_split'] = min_samples_split
                rfr_model_dict['Optimal Parameter']['max_depth'] = max_depth
                rfr_model_dict['Optimal Parameter']['ccp_alpha'] = ccp_alpha
                break

            dict = score_rank[i]
            rfc_final_model = RandomForestRegressor(criterion = dict['criterion'], min_samples_split = dict['min_samples_split'], max_depth = dict['max_depth'], ccp_alpha = dict['ccp_alpha'])

            rfr_model_dict['Optimal Parameter']['criterion'] = dict['criterion']
            rfr_model_dict['Optimal Parameter']['min_samples_split'] = dict['min_samples_split']
            rfr_model_dict['Optimal Parameter']['max_depth'] = dict['max_depth']
            rfr_model_dict['Optimal Parameter']['ccp_alpha'] = dict['ccp_alpha']

    # print(knn_performance_dict)
    # print(score_rank)

    for key in rfr_performance_dict:
        if rfr_performance_dict[key]['criterion'] == rfr_model_dict['Optimal Parameter']['criterion'] and rfr_performance_dict[key]['min_samples_split'] == rfr_model_dict['Optimal Parameter']['min_samples_split'] and rfr_performance_dict[key]['ccp_alpha'] == rfr_model_dict['Optimal Parameter']['ccp_alpha']:
            rfr_performance_dict[key]['flag'] = 1
            break
    
    rfr_model_dict['accuracy_record'] = rfr_performance_dict.copy()
    rfr_model_dict['optimal model'] = rfc_final_model
    final_model_dict['RFR'] = rfr_model_dict.copy()

    x.updateProgress(40,'Random Forest Regressor Regression Modelling Done')
    return True

def KNNRegressionModel(data_dict, model_dict, final_model_dict, x):

    highest = 0

    knnr_model_dict = model_dict['KNNR']

    knnr_performance_dict = {
       
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

    for w in knnr_model_dict['Parameter List']['weights']:
        for n in knnr_model_dict['Parameter List']['p']:

            my_dict = {
                'weights' : w,
                'p' : n,
                'val_score' : [],
                'train_score' : [],
                'flag' : 0
            }

            for k in knnr_model_dict['Parameter List']['n_neighbors']:

                score_rank_dict = {
                    'val_score' : 0,
                    'weights' : 0,
                    'p' : 0,
                    'n_neighbors' :0
                }

                knnr = KNeighborsRegressor(n_neighbors = k, weights = w, p = n)
                knnr.fit(xtrain,ytrain)
                ypred_val = knnr.predict(xval)

                # print(ypred_val)

                train_score = knnr.score(xtrain,ytrain)
                val_score = r2_score(yval, ypred_val)

                my_dict['val_score'].append(val_score)
                my_dict['train_score'].append(train_score)

                print('train: ',train_score,'   validate_score: ',val_score)

                goodModel = abs(train_score - val_score) / max(train_score, val_score) * 100 < overfitThreshold

                if goodModel:
                    score_rank_dict['val_score'] = val_score
                    score_rank_dict['weights'] = w
                    score_rank_dict['p'] = n
                    score_rank_dict['n_neighbors'] = k

                    score_rank.append(score_rank_dict.copy())

                if val_score > highest:
                    if goodModel:
                        knnr_model_dict['Optimal Parameter']['weights'] = w
                        knnr_model_dict['Optimal Parameter']['n_neighbors'] = k
                        knnr_model_dict['Optimal Parameter']['p'] = n
                        goodModelCount += 1

                    elif goodModel == False and goodModelCount == 0:
                        knnr_model_dict['Optimal Parameter']['weights'] = w
                        knnr_model_dict['Optimal Parameter']['n_neighbors'] = k
                        knnr_model_dict['Optimal Parameter']['p'] = n
                        
                    highest = val_score
                    highest_train = train_score

            knnr_performance_dict[id] = my_dict.copy()
            id += 1

    score_rank = sorted(score_rank, key=lambda x: x['val_score'], reverse = True)
    
    validModel = False
    i = 0
    # knn_final_model = knn_model_dict['Optimal Model']

    weights=knnr_model_dict['Optimal Parameter']['weights']
    p=knnr_model_dict['Optimal Parameter']['p']
    n_neighbors=knnr_model_dict['Optimal Parameter']['n_neighbors']

    # knnr_final_model = KNeighborsRegressor(weights=knnr_model_dict['Optimal Parameter']['weights'], p=knnr_model_dict['Optimal Parameter']['p'], n_neighbors=knnr_model_dict['Optimal Parameter']['n_neighbors'] )
    knnr_final_model = KNeighborsRegressor(weights=weights, p = p, n_neighbors=n_neighbors)
    xtrainFinal = data_dict['trainActual']['x']
    ytrainFinal = data_dict['trainActual']['y']

    scoreDictLength = len(score_rank)

    while validModel == False:

        knnr_final_model.fit(xtrainFinal,ytrainFinal)
        ypred = knnr_final_model.predict(xtest)

        print(highest)
        print(highest_train)

        test_score = r2_score(ytest,ypred)
        mse = mean_squared_error(ytest, ypred)

        print(test_score)

        goodFinalModel = abs(test_score - highest) / max(test_score, highest) * 100 < overfitThreshold

        # print(precision_score(ytest,ypred))
        # print(recall_score(ytest,ypred))

        print(ypred)

        if goodFinalModel:

            knnr_model_dict['r2_score'] = round(test_score,2)
            knnr_model_dict['Mean Squared Error'] = round(mse,2)
            # knn_model_dict['Optimal Model'] = json.dumps(knn_final_model)

            validModel = True

        else:
            i += 1

            if i >= scoreDictLength:
                knnr_final_model = KNeighborsRegressor(weights=weights, p = p, n_neighbors=n_neighbors)
                knnr_final_model.fit(xtrainFinal,ytrainFinal)
                ypred = knnr_final_model.predict(xtest)
                test_score = r2_score(ytest,ypred)
                mse = mean_squared_error(ytest, ypred)
                knnr_model_dict['r2_score'] = round(test_score,2)
                knnr_model_dict['Mean Squared Error'] = round(mse,2)

                knnr_model_dict['Optimal Parameter']['weights'] = weights
                knnr_model_dict['Optimal Parameter']['n_neighbors'] = n_neighbors
                knnr_model_dict['Optimal Parameter']['p'] = p
                break

            dict = score_rank[i]
            knnr_final_model = KNeighborsRegressor(weights = dict['weights'], p = dict['p'], n_neighbors = dict['n_neighbors'])

            knnr_model_dict['Optimal Parameter']['weights'] = dict['weights']
            knnr_model_dict['Optimal Parameter']['n_neighbors'] = dict['n_neighbors']
            knnr_model_dict['Optimal Parameter']['p'] = dict['p']

    # print(knn_performance_dict)
    # print(score_rank)

    for key in knnr_performance_dict:
        if knnr_performance_dict[key]['weights'] == knnr_model_dict['Optimal Parameter']['weights'] and knnr_performance_dict[key]['p'] == knnr_model_dict['Optimal Parameter']['p']:
            knnr_performance_dict[key]['flag'] = 1
            break
    
    knnr_model_dict['accuracy_record'] = knnr_performance_dict.copy()
    knnr_model_dict['optimal model'] = knnr_final_model
    final_model_dict['KNNR'] = knnr_model_dict.copy()

    x.updateProgress(40,'K-Nearest Neighbors Regression Modelling Done')
    print('KNNR')
    return True

def SVMRegressionModel(data_dict, model_dict, final_model_dict, x):

    highest = 0
    highest_train = 0

    svr_model_dict = model_dict['SVR']

    svr_performance_dict = {
       
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

    for kernel in svr_model_dict['Parameter List']['kernel']:
        for gamma in svr_model_dict['Parameter List']['gamma']:

            my_dict = {
                'kernel' : kernel,
                'gamma' : gamma,
                'val_score' : [],
                'train_score' : [],
                'flag' : 0
            }

            for C in svr_model_dict['Parameter List']['C']:

                score_rank_dict = {
                    'val_score' : 0,
                    'kernel' : 0,
                    'gamma' : 0,
                    'C' :0
                }

                svr = SVR(C = C, kernel = kernel, gamma = gamma)
                svr.fit(xtrain,ytrain)
                ypred_val = svr.predict(xval)

                # print(ypred_val)

                train_score = svr.score(xtrain,ytrain)
                val_score = r2_score(yval, ypred_val)

                my_dict['val_score'].append(val_score)
                my_dict['train_score'].append(train_score)

                print('train: ',train_score,'   validate_score: ',val_score)

                goodModel = abs(train_score - val_score) / max(train_score, val_score) * 100 < overfitThreshold

                if goodModel:
                    score_rank_dict['val_score'] = val_score
                    score_rank_dict['kernel'] = kernel
                    score_rank_dict['gamma'] = gamma
                    score_rank_dict['C'] = C

                    score_rank.append(score_rank_dict.copy())

                if val_score > highest:
                    if goodModel:
                        svr_model_dict['Optimal Parameter']['kernel'] = kernel
                        svr_model_dict['Optimal Parameter']['C'] = C
                        svr_model_dict['Optimal Parameter']['gamma'] = gamma
                        goodModelCount += 1

                    elif goodModel == False and goodModelCount == 0:
                        svr_model_dict['Optimal Parameter']['kernel'] = kernel
                        svr_model_dict['Optimal Parameter']['C'] = C
                        svr_model_dict['Optimal Parameter']['gamma'] = gamma
                        
                    # knn_model_dict['Optimal Model'] = knn
                    # knn_model_dict['Accuracy'] = val_score
                    # knn_model_dict['Precision'] = precision_score(ytest,ypred)
                    # knn_model_dict['Recall'] = recall_score(ytest,ypred)
                    highest = val_score
                    highest_train = train_score

                count += 1
                print(count)
            svr_performance_dict[id] = my_dict.copy()
            id += 1

    score_rank = sorted(score_rank, key=lambda x: x['val_score'], reverse = True)

    print(highest)
    print(highest_train)

    validModel = False
    i = 0
    # knn_final_model = knn_model_dict['Optimal Model']

    C=svr_model_dict['Optimal Parameter']['C']
    kernel=svr_model_dict['Optimal Parameter']['kernel']
    gamma=svr_model_dict['Optimal Parameter']['gamma']

    # svr_final_model = SVR(C=svr_model_dict['Optimal Parameter']['C'], kernel=svr_model_dict['Optimal Parameter']['kernel'], gamma=svr_model_dict['Optimal Parameter']['gamma'] )
    svr_final_model = SVR(C = C, kernel = kernel, gamma = gamma)
    xtrainFinal = data_dict['trainActual']['x']
    ytrainFinal = data_dict['trainActual']['y']

    scoreDictLength = len(score_rank)

    while validModel == False:

        svr_final_model.fit(xtrainFinal,ytrainFinal)
        ypred = svr_final_model.predict(xtest)

        print(highest)
        print(highest_train)

        test_score = r2_score(ytest,ypred)
        mse = mean_squared_error(ytest,ypred)

        print(test_score)

        goodFinalModel = abs(test_score - highest) / max(test_score, highest) * 100 < overfitThreshold

        # print(precision_score(ytest,ypred))
        # print(recall_score(ytest,ypred))

        print(ypred)

        if goodFinalModel:

            svr_model_dict['r2_score'] = round(test_score,2)
            svr_model_dict['Mean Squared Error'] = round(mse,2)

            # knn_model_dict['Optimal Model'] = json.dumps(knn_final_model)

            validModel = True

        else:
            i += 1

            if i >= scoreDictLength:
                svr_final_model = SVR(C = C, kernel = kernel, gamma = gamma)
                svr_final_model.fit(xtrainFinal,ytrainFinal)
                ypred = svr_final_model.predict(xtest)
                test_score = r2_score(ytest,ypred)
                mse = mean_squared_error(ytest,ypred)
                svr_model_dict['r2_score'] = round(test_score,2)
                svr_model_dict['Mean Squared Error'] = round(mse,2)

                svr_model_dict['Optimal Parameter']['kernel'] = kernel
                svr_model_dict['Optimal Parameter']['gamma'] = gamma
                svr_model_dict['Optimal Parameter']['C'] = C
                break

            dict = score_rank[i]
            svr_final_model = SVR(C = dict['C'], kernel = dict['kernel'], gamma = dict['gamma'])

            svr_model_dict['Optimal Parameter']['kernel'] = dict['kernel']
            svr_model_dict['Optimal Parameter']['gamma'] = dict['gamma']
            svr_model_dict['Optimal Parameter']['C'] = dict['C']

    # print(knn_performance_dict)
    # print(score_rank)

    for key in svr_performance_dict:
        if svr_performance_dict[key]['kernel'] == svr_model_dict['Optimal Parameter']['kernel'] and svr_performance_dict[key]['gamma'] == svr_model_dict['Optimal Parameter']['gamma']:
            svr_performance_dict[key]['flag'] = 1
            break
    
    svr_model_dict['accuracy_record'] = svr_performance_dict.copy()
    svr_model_dict['optimal model'] = svr_final_model
    final_model_dict['SVR'] = svr_model_dict.copy()

    x.updateProgress(40,'Support Vector Machine Regression Modelling Done')
    return True

def PLRegressionModel(data_dict, model_dict, final_model_dict, x):

    highest = 0

    plr_model_dict = model_dict['PLR']

    plr_performance_dict = {
       
    }

    score_rank = []

    xtrain = data_dict['train']['x']
    ytrain = data_dict['train']['y']

    xval = data_dict['validation']['x']
    yval = data_dict['validation']['y']

    xtest = data_dict['testing']['x']
    ytest = data_dict['testing']['y']

    xtrainFinal = data_dict['trainActual']['x']
    ytrainFinal = data_dict['trainActual']['y']

    id = 0
    goodModelCount = 0

    lr = LinearRegression()

    my_dict = {
        'degree' : [1,2,3,4,5,6],
        'val_score' : [],
        'train_score' : [],
        'flag' : 1
    }

    for degree in plr_model_dict['Parameter List']['degree']:

        score_rank_dict = {
            'degree' : 1,
            'val_score' : 0,
        }

        poly = PolynomialFeatures(degree=degree)

        xtrain_poly = poly.fit_transform(xtrain.copy())
        xval_poly = poly.transform(xval.copy())

        lr.fit(xtrain_poly, ytrain)
        ypred_val = lr.predict(xval_poly)

        train_score = lr.score(xtrain_poly,ytrain)
        val_score = r2_score(yval, ypred_val)

        my_dict['val_score'].append(val_score)
        my_dict['train_score'].append(train_score)

        goodModel = abs(train_score - val_score) / max(train_score, val_score) * 100 < overfitThreshold

        if goodModel:
            score_rank_dict['val_score'] = val_score
            score_rank_dict['degree'] = degree
            score_rank.append(score_rank_dict.copy())

        if val_score > highest:
            if goodModel:
                plr_model_dict['Optimal Parameter']['degree'] = degree
                goodModelCount += 1

            elif goodModel == False and goodModelCount == 0:
                plr_model_dict['Optimal Parameter']['degree'] = degree

            highest = val_score

    plr_performance_dict[id] = my_dict.copy()
    id += 1

    score_rank = sorted(score_rank, key=lambda x: x['val_score'], reverse = True)
    print(score_rank)

    validModel = False
    i = 0
    degree = plr_model_dict['Optimal Parameter']['degree']

    poly = PolynomialFeatures(degree=degree)
    xtrainFinal_poly = poly.fit_transform(xtrainFinal.copy())
    xtest_poly = poly.transform(xtest.copy())

    lr_final_model = LinearRegression()

    scoreDictLength = len(score_rank)

    while validModel == False:

        lr_final_model.fit(xtrainFinal_poly, ytrainFinal)
        ypred = lr_final_model.predict(xtest_poly)

        test_score = r2_score(ytest,ypred)
        mse = mean_squared_error(ytest,ypred)

        goodFinalModel = abs(test_score - highest) / max(test_score, highest) * 100 < overfitThreshold
        if goodFinalModel:
            plr_model_dict['r2_score'] = round(test_score,2)
            plr_model_dict['Mean Squared Error'] = round(mse,2)
            validModel = True

        else:
            i += 1

            if i >= scoreDictLength:
                poly = PolynomialFeatures(degree=degree)
                xtrainFinal_poly = poly.fit_transform(xtrainFinal.copy())
                xtest_poly = poly.transform(xtest.copy())
                lr_final_model.fit(xtrainFinal_poly, ytrainFinal)
                ypred = lr_final_model.predict(xtest_poly)
                test_score = r2_score(ytest,ypred)
                mse = mean_squared_error(ytest,ypred)
                plr_model_dict['r2_score'] = round(test_score,2)
                plr_model_dict['Mean Squared Error'] = round(mse,2)

                plr_model_dict['Optimal Parameter']['degree'] = degree
                break

            dict = score_rank[i]
  
            poly = PolynomialFeatures(degree=dict['degree'])
            xtrainFinal_poly = poly.fit_transform(xtrainFinal.copy())
            xtest_poly = poly.transform(xtest.copy())

            plr_model_dict['Optimal Parameter']['degree'] = dict['degree']

    # for key in plr_performance_dict:
    #     if (plr_performance_dict[key]['degree'] == plr_model_dict['Optimal Parameter']['degree']):
    #         plr_performance_dict[key]['flag'] = 1
    #         break

    plr_model_dict['accuracy_record'] = plr_performance_dict.copy()
    plr_model_dict['optimal model'] = poly
    final_model_dict['PLR'] = plr_model_dict.copy()

    x.updateProgress(40,'Polynomial Regression Modelling Done')
    print('PLR')
    return True

def regression(df_modelling_dict, model_choice, x, modelType):

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

    dataSplitting(df_modelling_dict, target, data_dict)

    model_dict = {
        'LR' : {
            'Name':'Linear Regression',
            'Parameter List' : {
                'fit_intercept' : [False, True],
                'normalize' : [False, True]
            },
            'Optimal Parameter' : {
                'fit_intercept' : True,
                'normalize' : False
            },
            'r2_score' : 0,
            'Mean Squared Error' :0,
            'main_axis' : {
                'value' : [],
                'name' : ''
            }
        },

        'RFR' :{
            'Name':'Random Forest Regressor',
            'Parameter List' :{
                'max_depth' : [n for n in range(2,20,2)],
                'criterion' : ['squared_error', 'absolute_error', 'poisson'],
                'min_samples_split' : [2,3,4],
                'ccp_alpha' : [0, 0.01, 0.05, 0.1]
            },

            'Optimal Parameter' : {
                'max_depth' : 0,
                'criterion' : 0,
                'min_samples_split' : 0,
                'ccp_alpha' :0
            },

            
            'r2_score': 0,
            'Mean Squared Error' : 0,
            'main_axis' : {
                'value':[n for n in range(2,20,2)],
                'name' : 'max_depth'
                }
        },

        'DTR' :{
            'Name':'Decision Tree Regressor',
            'Parameter List' :{
                'max_depth' : [n for n in range(2,20,2)],
                'criterion' : ['squared_error', 'friedman_mse', 'absolute_error','poisson'],
                'min_samples_split' : [2,3,4],
                'ccp_alpha' : [0, 0.01, 0.05, 0.1]
            },

            'Optimal Parameter' : {
                'max_depth' : 0,
                'criterion' : 0,
                'min_samples_split' : 0,
                'ccp_alpha' :0
            },
           
            'r2_score': 0,
            'Mean Squared Error' : 0,
            'main_axis' : {
                'value':[n for n in range(2,20,2)],
                'name' : 'max_depth'
                }
        },

        'KNNR' :{
            'Name':'K-Nearest Neighbour Regressor',
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
            
            'r2_score': 0,
            'Mean Squared Error' : 0,
            'main_axis' : {
                'value':[k for k in range(3,40,2)],
                'name' : 'n_neighbors'
                } 
        },

        'SVR' : {
            'Name':'Support Vector Machine Regressor',
            'Parameter List' : {
                'C' : [0.01, 0.1, 1],
                'kernel' : ['linear', 'rbf', 'sigmoid'],
                'gamma' : ['scale', 'auto']
            },

            'Optimal Parameter' : {
                'C' : 0.01,
                'kernel' : 'linear',
                'gamma' : 'auto'
            },

            'r2_score': 0,
            'Mean Squared Error' : 0,
            'main_axis' : {
                'value':[0.01, 0.1, 1],
                'name' : 'C'
                }
        },

        'PLR' : {
            'Name':'Polynomial Regression',
            'Parameter List' : {
                'degree' : [1,2,3,4,5,6]
            },

            'Optimal Parameter' : {
                'degree' : 1
            },

            'r2_score': 0,
            'Mean Squared Error' : 0,
            'main_axis' : {
                'value':[1,2,3,4,5,6],
                'name' : 'degree'
                }
        }
    }

    final_model_dict = {
        'modelType' : modelType,
        'model' : {}
    }

    func_dict = {
        'LR': linearRegressionModel,
        'RFR': RFRegressionModel,
        'DTR' : DTRegressionModel,
        'KNNR' : KNNRegressionModel,
        'SVR' : SVMRegressionModel,
        'PLR' : PLRegressionModel
    }

    for model in func_dict:
        if model in model_choice:
            # func_dict[model](data_dict, model_dict, final_model_dict['model'], x)
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
        print('R2_score: ',model_dict[key]['r2_score'])
        print('MSE: ', model_dict[key]['Mean Squared Error'])
        print('Optimal Parameter: ', model_dict[key]['Optimal Parameter'])

    # Convert boolean to string to avoid formatting problem in later process
    if 'LR' in model_choice:
        a = final_model_dict['model']['LR']
        a['Optimal Parameter']['fit_intercept'] = str(a['Optimal Parameter']['fit_intercept'])
        a['Optimal Parameter']['normalize'] = str(a['Optimal Parameter']['normalize'])

        for key in a['Parameter List']:
            i = 0
            for val in a['Parameter List'][key]:
                a['Parameter List'][key][i] = str(val)
                i += 1


    return final_model_dict

