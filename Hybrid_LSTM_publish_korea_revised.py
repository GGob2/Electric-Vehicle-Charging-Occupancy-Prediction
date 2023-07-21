# -*- coding: utf-8 -*-
"""
Created on Nov 2, 2021
Code for charging station occupancy prediction using hybrid LSTM 
and Machine Learning approaches.
More detail: see the paper: 
Ma, TY, Faye, S. (2021) Multistep Electric Vehicle Charging Station Occupancy
 Prediction using Hybrid LSTM Neural Networks. arXiv:2106.04986
    
@author: Tai-yu MA
"""

import math
import time
import pandas as pd
import numpy  as np
import graphviz
import time

from numpy import array
from matplotlib import pyplot
from tensorflow import keras 
from keras.models import Sequential 
from keras.layers import LSTM 
from keras.layers import Dense,Dropout  
from keras.models import Model 
from keras.layers import Input
from keras.layers import concatenate

from sklearn import model_selection
from sklearn.linear_model import LogisticRegression 
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier 
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV

from xgboost import XGBClassifier


## read data for machine learning approaches
def read_data_ML(string):
    Z = pd.read_csv(string)
    Z=Z.to_numpy()
     
    n_train=int(0.7*len(Z))         # 훈련 데이터 비율
    
    X_train=Z[0: n_train,0:-1];         y_train = Z[0:n_train,-1]
    X_test =Z[n_train: len(Z),0:-1];    y_test  = Z[n_train:len(Z),-1]
    
    return X_train,y_train,X_test,y_test 

# split a multivariate sequence into samples
def split_sequences(sequences, n_steps_in, n_steps_out): 
	X, y = list(), list()
	for i in range(len(sequences)):
		# find the end of this pattern
		end_ix = i + n_steps_in                 # 입력 시퀀스의 끝 인덱스
		out_end_ix = end_ix + n_steps_out-1     # 출력 시퀀스의 끝 인덱스
		# check if we are beyond the dataset
		if out_end_ix > len(sequences):         # 인덱스를 벗어나지 않도록 함
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1 : out_end_ix, -1]     # 시퀀스 생성
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)

## 하이브리드 LSTM을 위한 데이터 읽어오기
def read_data(string, string2, model_id, n_steps_in, n_steps_out, n_features):
    print('read data 함수 실행') 
    Z = pd.read_csv(string)       # 파일 읽어오기 col1 time, col2 
    Z = Z.to_numpy()              # numpy로 변환
    
    X, y = split_sequences(Z[:,3:5], n_steps_in, n_steps_out )
    
    n_train = int(0.7*len(X))   # 학습 비율 설정
    Z1 = pd.read_csv(string2)   # col1 weekday, col2 weekend(string2 = 평균 충전 점유)
    Z1 = Z1.to_numpy()
    Z1 = Z1.transpose()
    Z2 = np.concatenate((Z1,Z1),axis=1) # 열 1개로 합치기
    X2 = np.zeros([len(Z),3+144],float)  
  
    for i in range(len(Z) - n_steps_in): 
     if Z[i+n_steps_in-1,-1] == 0:      
          qq    = np.array(Z2[0][0:144]) # weekday 평균 점유율 벡터
          X2[i] = np.append(Z[i+n_steps_in-1][0:3], qq) 
     else:            
         qq    = np.array(Z2[1][0:144])  # weekend 평균 점유율 벡터
         X2[i] = np.append(Z[i+n_steps_in-1][0:3], qq) 
    
    X_train = X[0: n_train,];         y_train = y[0:n_train,]
    X_test  = X[n_train: len(X),];    y_test  = y[n_train:len(X),]
   
    X2_train = X2[0: n_train,]; X2_test = X2[n_train: len(X),];  
   
    
    return X_train, y_train, X_test, y_test, X2_train, X2_test
 
    
############
# Mix_LSTM
############
def fit_model_MixLSTM(res_F1, res ,_iter, X_train, y_train, X_test, y_test, X2_train, X2_test,
                   n_steps_in, n_steps_out, n_features, n_n_lstm, dropout, n_epoch, bat_size):
    
    input1 = keras.Input(shape=(n_steps_in, n_features))
    input2 = keras.Input(shape=(147,))

    input1_reshaped = keras.layers.Reshape((n_steps_in, 1, 1, n_features))(input1)

    # CNN-LSTM
    model_CNN = keras.layers.Conv1D(filters=16, kernel_size=3, activation='relu')(input1)
    model_CNN = keras.layers.MaxPooling1D(pool_size=2)(model_CNN)
    

    # ConvLSTM 변경
    #model_ConvLSTM = keras.layers.ConvLSTM2D(n_n_lstm, (1, 1))(input1_reshaped)
    #model_ConvLSTM = keras.layers.Flatten()(model_ConvLSTM)
    #model_ConvLSTM = keras.layers.Dropout(dropout)(model_ConvLSTM)
    #model_ConvLSTM = keras.layers.Dense(18, activation='relu')(model_ConvLSTM)


    #model_LSTM=LSTM(n_n_lstm)(input1)
    model_LSTM=LSTM(n_n_lstm)(model_CNN)
    model_LSTM=Dropout(dropout)(model_LSTM)
    model_LSTM=Dense(18, activation='relu')(model_LSTM)
   
    meta_layer = keras.layers.Dense(147, activation="relu")(input2)
    meta_layer = keras.layers.Dense(64, activation="relu")(meta_layer)    
    meta_layer = keras.layers.Dense(32, activation="relu")(meta_layer)

    model_merge = keras.layers.concatenate([model_LSTM, meta_layer])
    model_merge = Dense(100, activation='relu')(model_merge)
    model_merge = Dropout(dropout)(model_merge)    
    output = Dense(n_steps_out, activation='sigmoid')(model_merge)
    model = Model(inputs=[input1, input2], outputs=output) 
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) 
    model.fit([X_train, X2_train], y_train, epochs=n_epoch, batch_size=bat_size, verbose=2) # verbose=2는 함축적인 정보 출력, y_train은 정답 데이터
    
    temp = model.predict([X_test, X2_test], verbose=2)
    
    m, n = temp.shape 
    t_target = n_steps_out

    # precision, recall, f1_score를 위한 값들
    yhat = np.zeros((m, t_target))
    y_obs = np.array(y_test[0:m, 0:t_target])
    scores1 = np.zeros(m)
    scores_F1 = np.zeros([m,3],float)

    for i in np.arange(m):  
        for j in np.arange(t_target):  
            if temp[i][j]>=0.5:
                yhat[i][j]=1           
        
        val = 1 - sum(abs(yhat[i,]-y_obs[i,:])) / t_target

        scores_F1[i,0] = precision_score(y_obs[i,:], yhat[i,], zero_division=1)
        scores_F1[i,1] = recall_score(y_obs[i,:], yhat[i,], zero_division=1)
        scores_F1[i,2] = f1_score(y_obs[i,:], yhat[i,], zero_division=1)     # zero_division=1 인 경우, 0으로 나누는 오류가 발생할 때, 값을 1로 넣음
        scores1[i] = val       
     
    _mean1 = np.mean(scores1)      
    _mean_F1 = np.mean(scores_F1, axis=0)  
    res[_iter, :] = [n_n_lstm, dropout, n_epoch, bat_size, _mean1]  
    res_F1[_iter, :]= _mean_F1
    return res_F1, res     

def fit_model_DecisionTree(X_train, y_train, X_test, y_test, X2_train, X2_test):
    
    #X_train_combined = X_train.reshape(X_train.shape[0], -1)
    #X_test_combined = X_test.reshape(X_test.shape[0], -1)
    
    # # 메타 특성을 합치기 위해 입력 데이터를 결합합니다.
    X_train_combined = np.hstack((X_train.reshape(X_train.shape[0], -1), X2_train))
    X_test_combined = np.hstack((X_test.reshape(X_test.shape[0], -1), X2_test))

    # 결정 트리 분류기를 정의하고, 학습 데이터를 사용하여 모델을 학습시킵니다.
    dt_classifier = DecisionTreeClassifier(random_state=42)
    dt_classifier.fit(X_train_combined, y_train)

    important_feature =  dt_classifier.feature_importances_

    # 테스트 데이터를 사용하여 예측을 수행합니다.
    y_pred = dt_classifier.predict(X_test_combined)

    # 성능 평가 지표를 계산합니다.
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=1)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=1)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=1)
    
    res_list = [acc, precision, recall, f1]
    

    # print("Accuracy: ", acc)
    # print("Precision: ", precision)
    # print("Recall: ", recall)
    # print("F1-score: ", f1)

    return res_list, important_feature


def fit_model_xgboost(X_train, y_train, X_test, y_test, X2_train, X2_test):
    
    X_train_combined = np.hstack((X_train.reshape(X_train.shape[0], -1), X2_train))
    X_test_combined = np.hstack((X_test.reshape(X_test.shape[0], -1), X2_test))
    
    #X_train_combined = X_train.reshape(X_train.shape[0], -1)
    #X_test_combined = X_test.reshape(X_test.shape[0], -1)

    # XGBoost classifier를 초기화합니다.
    model = XGBClassifier(n_estimators=500, learning_rate=0.2, max_depth=4, random_state=42)
    model.fit(X_train_combined, y_train,  verbose=True)
    
    
    # GridSearchCV를 위한 하이퍼파라미터 그리드를 생성합니다.
    # param_grid = {
    #     'n_estimators': [100, 200, 300, 400, 500, 50, 150],
    #     'learning_rate': [0.01, 0.05, 0.1, 0.07, 0.04, 0.09]
    # }

    # # GridSearchCV를 초기화합니다.
    #grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)

    # # GridSearchCV를 이용하여 최적의 하이퍼파라미터를 찾고 모델을 학습시킵니다.
    #grid_search.fit(X_train_combined, y_train)

    # # 최적의 하이퍼파라미터를 출력합니다.
    # print(f"Best Parameters: {grid_search.best_params_}")

    # 최적의 모델로 부터 예측을 얻습니다.
    y_pred = model.predict(X_test_combined)

    # 성능 평가를 위한 정확도, 정밀도, 재현율, F1-score를 계산합니다.
    accuracy = accuracy_score(y_test, y_pred)
    # precision = precision_score(y_test, y_pred, zero_division=1)
    # recall = recall_score(y_test, y_pred, zero_division=1)
    # f1 = f1_score(y_test, y_pred, zero_division=1)
    
    #res_list = [accuracy, precision, recall, f1]

    return accuracy

# 머신러닝 모델 구현을 위한 코드
def run_ML(model_id,n_steps_out):
     
    n_station=9 
    string='ML/data_chg_ML_'
    station=[string+'1.csv',string+'2.csv',string+'3.csv',string+'4.csv',string+'5.csv',string+'6.csv',string+'7.csv',string+'8.csv',string+'9.csv']

    if    model_id=='logistic': idx_model = LogisticRegression()
    elif  model_id=='svc':      idx_model = SVC()
    elif  model_id=='RF':       idx_model = RandomForestClassifier() 
    elif  model_id=='Ada':      idx_model = AdaBoostClassifier() 
    
    models=[idx_model]
    rng_s_1=[s for s in range(3)]    
    rng_s_2=[s for s in range(6)]
    rng_s_3=[s for s in range(12)]
    rng_s_4=[s for s in range(24)]
    rng_s_5=[s for s in range(36)]
     
    step_set=[rng_s_1,rng_s_2,rng_s_3,rng_s_4,rng_s_5] 
    
    vec_mean_metrics=[]

    for s in range(n_station):         
        X_train,y_train,X_test,y_test =read_data_ML(station[s])     # 데이터 불러오기
        res_all=[] #activate for multi-steps
        for mm in models:
            mm.fit(X_train, y_train)                                # 모델 학습 수행
           
            t_target = n_steps_out
            
            m,n=X_test.shape
            yhat=np.zeros([m,t_target])            
            y_obs=np.zeros([m,t_target])
            
            for kk in range(m-t_target) :
                y_obs[kk,:]=y_test[kk:kk+t_target]
            
            n_sample=m-n_steps_out

            for i in range(n_sample): 
                X_test_temp=X_test.copy();  
                X_test_temp=np.append(X_test_temp,[[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],],0) #dummy
                
                for j in range(n_steps_out):   
                    temp11=X_test_temp[i+j,:].reshape(1, -1)                    
                    yhat[i,j] = mm.predict(temp11) 
                    
                    rng1=[ i+j+3 -_ii for _ii in range(0, 3) ]
                    rng2=[ _jj for _jj in range(-3,0) ] # only t-3,-2,-1 are considered
                    X_test_temp[rng1,rng2]=yhat[i,j]  
                  
                res_temp=[]
                for rr in step_set: 
                    _acc   = accuracy_score(y_obs[i,rr], yhat[i,rr])
                    _pre   =precision_score(y_obs[i,rr], yhat[i,rr],zero_division=1)
                    _recall= recall_score(y_obs[i,rr], yhat[i,rr],zero_division=1)
                    _f1    = f1_score(y_obs[i,rr], yhat[i,rr],zero_division=1)
                    res_temp=np.append(res_temp,[_acc, _pre,_recall,_f1],0)
                
                res_all.append(res_temp) 
                
            _mean_metrics = np.mean(res_all,axis=0)   
            vec_mean_metrics.append(_mean_metrics)
            
    return  vec_mean_metrics
    
def run(model_id, n_steps_in, n_steps_out, n_features, n_epoch, n_trivals, n_out,
            n_n_lstm, dropout, bat_size): 
    duration_list = []
    
    
    print('run 함수 실행')
            
    
    #n_station = 9

    #string =   'mixed_LSTM/data_chg_'
    #string2 =  'mixed_LSTm/data_chg_pred_occ_t_'
    
    #station =  [string+'1.csv',string+'2.csv',string+'3.csv',string+'4.csv',string+'5.csv',string+'6.csv',string+'7.csv', string+'8.csv', string+'9.csv']
    #station2 = [string2+'1.csv',string2+'2.csv',string2+'3.csv',string2+'4.csv',string2+'5.csv',string2+'6.csv',string2+'7.csv', string2+'8.csv', string2+'9.csv']
    
    n_station = 7
    
    string =   'korea_data/data_chg_'
    string2 =  'korea_data/data_chg_pred_occ_t_'

    station =  [string+'1.csv',string+'2.csv',string+'3.csv',string+'4.csv',string+'5.csv',string+'6.csv',string+'7.csv']
    station2 = [string2+'1.csv',string2+'2.csv',string2+'3.csv',string2+'4.csv',string2+'5.csv',string2+'6.csv',string2+'7.csv']

    
  
    res_all=[]; res_all_F1=[]
    
    for s in range(n_station):         
        start_time = time.time()
        
        X_train, y_train, X_test, y_test, X2_train, X2_test = read_data(station[s], station2[s], model_id,  
                                n_steps_in, n_steps_out, n_features)
        
        
        res = np.zeros([n_trivals, n_out])
        res_F1 = np.zeros([n_trivals,3])
        
        for _iter in range(n_trivals):   
            if model_id == 'Mix_LSTM': # 전해진 파라미터가 Mix_LSTM
                 res_F1,res = fit_model_MixLSTM(res_F1, res,_iter, X_train, y_train, X_test, y_test, X2_train, X2_test, 
                           n_steps_in, n_steps_out, n_features, n_n_lstm, dropout, n_epoch, bat_size)
            elif model_id == 'decision_tree':
                res_list, important_features = fit_model_DecisionTree(X_train, y_train, X_test, y_test, X2_train, X2_test)
            elif model_id == 'XGBoost':
                 res_list = fit_model_xgboost(X_train, y_train, X_test, y_test, X2_train, X2_test)
        end_time = time.time()
       
        duration = end_time - start_time
        duration_list.append(duration)
        print("학습 소요 시간 : ", duration)
       
        df = pd.DataFrame({'time': duration_list})
        df.to_csv('time_uk_cnnSTM.csv')

        #print(res_F1)
        #print(res)
        
        #print('acc : ',  res_list) 
        #print('\nfeatures_importances_ : \n', important_features)
        
         
        _mean = np.mean(res[:,-1:], axis=0)
        _std  = np.std(res[:,-1:], axis=0)
        res_all.append([_mean,_std])
       
        _mean_F1 = np.mean(res_F1, axis=0)
        res_all_F1.append([_mean_F1])
        
    temp=[]
    for i in range(n_station):            
        temp.append(res_all[i][0])

        accuracy_avg1=np.mean(temp, axis=0) # 전체 아웃풋에 대한 평균 값
    accuracy_avg2=np.mean(temp, axis=1)     # 아웃풋 각각을 저장
    avg_metrics_prec_recall_F1 = np.mean(res_all_F1, axis=0)
    
    return accuracy_avg1, accuracy_avg2, res_all, res_all_F1, avg_metrics_prec_recall_F1
    #return res_list


def main():
   
    n_steps_in = 12     # LSTM 셀을 위한 인풋 y 시퀀스 수 (2시간)
    n_features = 1      # LSTM 모델의 각 인풋 단계에서 고려할 시간 수 (1 = 10분) 대한 기능 수 지정
    n_steps_out = 6     # LSTM 모델 아웃풋에서 예측할 시간 수 (10분 단위 = 6개, 60분)
    n_epoch_global = 15   # 에포크 수 지정
    n_trivals = 10        # 학습 수행 수
    n_out = 5  
    n_n_lstm = 36         # LSTM 뉴런 수
    dropout = 0.2         # 드롭아웃 비율
    bat_size = 30        # 배치크기 지정
    accuracy_avg_1 = []
    accuracy_avg_2 = []
    flag_sensitivity = 0  
    model_id = 'Mix_LSTM' 
    flag_ML = 0  # to run machine learning models, set  flag_ML=1 otherwise 0
    # for ML, we set n_steps_out=36 as we compute the predcition for all forecasting cases 
    
    if  flag_ML == 1:
        n_steps_out = 36
    
    # 테스트할 머신러닝 모델 선택
    
    #model_id = 'logistic'
    #model_id = 'svc'
    #model_id = 'RF'
    #model_id = 'Ada'
    
    if flag_ML == 0:      # 머신러닝 모델 사용 안할 경우
        if flag_sensitivity == 1:  
            parameter = [11, 12] #,13,14,15,16,17,18,19,20 에포크 수 조절
            for i in range(len(parameter)):
                avg1, avg2, res_all, res_all_F1 = run(model_id, n_steps_in, n_steps_out, n_features,
                               parameter[i], n_trivals, n_out, n_n_lstm, dropout, bat_size)  
                accuracy_avg_1.append(avg1)
                accuracy_avg_2.append(avg2)            
        else:   
            avg1, avg2, res_all, res_all_F1, avg_metrics_prec_recall_F1 = run(model_id, n_steps_in, n_steps_out, n_features,
                                n_epoch_global, n_trivals, n_out, n_n_lstm, dropout, bat_size)  
            accuracy_avg_1.append(avg1) 
            accuracy_avg_2.append(avg2) 
            
        print('model: ', model_id)
        print('sensitivity_flag = ', flag_sensitivity) 
        if flag_sensitivity == 1:
            print('parameter : ', parameter) 
        print('n_step out: ', n_steps_out)
        print('n_epoch,n_trivals, n_n_lstm,dropout,bat_size', 
              n_epoch_global,n_trivals,n_n_lstm,dropout,bat_size)       
        print('accuracy_avg_1: ',accuracy_avg_1)
        print('accuracy_avg_2: ',accuracy_avg_2)
        
        df2 = pd.DataFrame({'acc': accuracy_avg_2})
        df2.to_csv('acc_uk_cnnLSTM.csv')

        print('avg_metrics_prec_recall_F1= ',avg_metrics_prec_recall_F1)
       
    else:        
        vec_mean_metrics = run_ML(model_id, n_steps_out)
        
        mean_all  = np.mean(vec_mean_metrics,axis=0)
        print('vec_mean_metrics',vec_mean_metrics)
        print('_mean_all',mean_all)
    
    #print(model_id + " 실행 결과")
    #result = run(model_id, n_steps_in, n_steps_out, n_features,
    #    n_epoch_global, n_trivals, n_out, n_n_lstm, dropout, bat_size)    
main() 


# precision = True로 분류한 것중, 실제 True의 비율
# recall = 실제 True인 것 중, 모델이 True로 분류한 데이터
# precision과 recall의 값이 모두 높아야 좋은 모델이라고 할 수 있음
# F1_score = precision과 recall의 조화 평균 (데이터 레이블이 불균형할 때 효과가 좋음)