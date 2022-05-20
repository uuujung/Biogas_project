# 데이터 표준화를 위한 파일.


import numpy as np
import pymysql
import pickle

import time


def IQR(data, axis): # Q3 - Q1
    return np.percentile(data, 75, axis) - np.percentile(data, 25, axis)


def normalize(X, location, scale): 
    return (X - location) / scale


def MAPE(y_true, y_pred): 
    return 1. - np.mean(np.abs((y_true - y_pred) / y_true), axis = 0)


def generate_data(data , n_in, n_key, n_out, example , gap, normalize, impute):

    def create_time_steps(length):
        return np.asarray(list(range(-length, 0))) / 100.
    
    
    #인수중에 impute를 False로 받으면 안쓰는 함수===============================
    def check_anormality_imputation(data, X, name): #정규성 대치 확인???????
        
        #mean = np.mean(X, axis = 0)
        
        median = np.median(data, axis = 0)
        iqr = IQR(data, 0)
        
        lower_fence = median - 1.5 * iqr
        upper_fence = median + 1.5 * iqr
        
        for i, x in enumerate(X):
            for j, xj in enumerate(x):
                if xj >= lower_fence[j] and xj <= upper_fence[j]:
                    pass
                else:
                    print('Anormaly dectected at %s(%d, %d): %.4f (%.4f, %.4f)' % (name, i, j, X[i, j], lower_fence[j], upper_fence[j]))
                    if xj <= lower_fence[j]:
                        X[i, j] = lower_fence[j]
                        print('Imputed by lower fence %.4f' % lower_fence[j])
                    if xj >= upper_fence[j]:
                        X[i, j] = upper_fence[j]
                        print('Imputed by upper fence %.4f' % upper_fence[j])
                    time.sleep(1)
        return X
    #=========================================================================
    
    # 데이터 가공 =============================================================
    data = data.values[:, 1:] # 열의 0번째가 time이라서 그거 제외하고 값만 데이터로 사용.
      
    len_data = len(data)

    X = np.reshape(create_time_steps(len_data), [len_data, 1]) #time_steps만든것을 [len_data , 1] 차원으로 리쉐잎
    #X = np.reshape(np.array(range(len_data)), [len_data, 1])
    X = X[-int(gap + example):-gap] #끝에서 gap+example 개 부터 gap + 1개 까지 즉 example개 나옴.
    #ex. X[-7:-5] -> 끝에서 7번째부터 끝에서 6번째 까지 즉 2개 나옴.

    X = np.append(X, data[-int(gap + example):-gap, :n_in], axis = 1) #x랑 data 열로 붙일것.
    #이때 data의 첫번째 열만 가져옴. 첫번째 열 값이 olr임.
    X = np.array(X, dtype = np.float32) #[time_steps, olr] 형태.
    
    Y1 = data[-int(example):, n_in:(n_in + n_key)] #끝에서 num_example개의 행만큼, 열은 키팩터만.
    Y2 = data[-int(example):, (n_in + n_key):] #열은 아웃풋만  #data[-int(example):, (n_in + n_key):] 
    #=========================================================================
    
    #impute = False 일 경우 계산 X ============================================
    if impute:
        Y1 = check_anormality_imputation(data[:,n_in:(n_in + n_key)], Y1, 'Y1')
        Y2 = check_anormality_imputation(data[:, (n_in + n_key):], Y2, 'Y2') #data[-int(example):, (n_in + n_key):]
    #=========================================================================
    
    Y1 = np.array(Y1, dtype = np.float32)
    Y2 = np.array(Y2, dtype = np.float32)

    #표준화 방법 선택==========================================================        
    if normalize == 'min-max':
        location_X = np.min(X, axis = 0)
        scale_X  = np.max(X, axis = 0) - np.min(X, axis = 0) 
        location_Y1 = np.min(Y1, axis = 0)
        scale_Y1  = np.max(Y1, axis = 0) - np.min(Y1, axis = 0)
        location_Y2 = np.min(Y2, axis = 0)
        scale_Y2  = np.max(Y2, axis = 0) - np.min(Y2, axis = 0)
    elif normalize == 'robust':
        location_X = np.median(X, axis = 0)
        scale_X  = IQR(X, 0)
        location_Y1 = np.median(Y1, axis = 0)
        scale_Y1  = IQR(Y1, 0)
        location_Y2 = np.median(Y2, axis = 0)
        scale_Y2  = IQR(Y2, 0)
    elif normalize == 'standardize':
        location_X = np.mean(X, axis = 0)
        scale_X  = np.std(X, axis = 0)
        location_Y1 = np.mean(Y1, axis = 0)
        scale_Y1  = np.std(Y1, axis = 0)
        location_Y2 = np.mean(Y2, axis = 0) 
        scale_Y2 = np.std(Y2, axis = 0) 
    else:
        location_X = np.min(X, axis = 0)
        scale_X  = np.max(X, axis = 0) - np.min(X, axis = 0) 
        location_Y1 = np.min(Y1, axis = 0)
        scale_Y1  = np.max(Y1, axis = 0) - np.min(Y1, axis = 0)
        location_Y2 = np.min(Y2, axis = 0)
        scale_Y2  = np.max(Y2, axis = 0) - np.min(Y2, axis = 0)
    #=========================================================================

    #선택된 방법으로 표준화====================================================
    X = (X - location_X) / scale_X
    X = np.asarray(X, np.float32)
    
    Y1 = (Y1 - location_Y1) / scale_Y1
    Y1 = np.asarray(Y1, np.float32)

    Y2 = (Y2 - location_Y2) / scale_Y2
    Y2 = np.asarray(Y2, np.float32)
    #=========================================================================
    return (X, Y1, Y2), ((location_X, location_Y1, location_Y2), (scale_X, scale_Y1, scale_Y2))
    # X :[time_steps, olr] Y1:[키팩터1,키팩터2,키팩터3], Y2:[아웃풋1, 아웃풋2] 이때 행은 모두

def generate_data_for_predict(data , stats, n_in, n_key):
    
    location, scale = stats
    location_X, location_Y1, location_Y2 = location
    scale_X, scale_Y1, scale_Y2 = scale
    #import datetime

    data = data.values[:, 1:] # 열의 0번째가 time이라서 그거 제외하고 값만 데이터로 사용.
    
    #start_time = datetime.datetime.strptime(str(data[0][0]), "%Y-%m-%d")
    #pred_time = datetime.datetime.strptime(str(data[-1][0]), "%Y-%m-%d")
    
    #test_X = np.reshape(np.array([int((pred_time - start_time).days)]), [1, 1])
    
    test_X = np.reshape(np.array([0.0]), [1, 1]) # 원래 X에 time_step있어서 그 열 맞춰주려고 0으로 열하나 생성하는 거 같음
    x = data[-1, :n_in] #제일 마지막 행의 n_in(1)번째 열 (olr)
    x = np.reshape(x, [1, len(x)])
    

    test_X = np.append(test_X, x, axis = 1)
    test_X = np.reshape(np.asarray(test_X, np.float32), [1, test_X.shape[1]]) #array([[0., 마지막olr]])
    test_X = (test_X - location_X) / scale_X

    
    test_Y1 = data[-1, n_in:(n_in + n_key)] #제일 마지막 행의 키팩터열들
    test_Y1 = (test_Y1 - location_Y1) / scale_Y1
    test_Y1 = np.reshape(np.asarray(test_Y1, np.float32), [1, len(test_Y1)])
    
    return (test_X, test_Y1) #학습에서 각각 pred_input, pred_key로 받음
    #Y2(output 데이터들)는 표준화 안시킴.
    #X의 타임을 임의로 0으로 해서 표준화 시켜도 학습에 지장없는지 궁금함.

    
