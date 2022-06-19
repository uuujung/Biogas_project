# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 21:15:49 2021

@author: ternery-lab0
"""


print('Start import tensorflow...', end = '')
import datetime
import numpy as np
import time    

import db_io_utils as utils
import Utils_online as Utils

import Networks as Networks

import tensorflow as tf
import pandas as pd

#from kde import KDE as kde_
import kde as KDE
import pickle
import matplotlib.pyplot as plt
from matplotlib import patches
#tf.random.set_seed(2020)
print('Done.')


gpu = 1
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(gpus[gpu], 'GPU')
tf.config.experimental.set_memory_growth(gpus[gpu], True)


# config 불러오기 =============================================================

# config(설정파일)에서 설정 할 수 있는것========================================
# 1. source(소스)
#   site, database, raw_data_table이름, 가공 data_table 이름, pred_table 이름
# 2. parameters (파라미터)
#   input, key, output 갯수 설정
#   gap, n_example, fien_example, batch_size
# 3. tuning_parameters
#   units 노드의 갯수, 활성화함수, 초기화 값 설정
# 4. vars (값 설정 하는건가?)
#   input ,key, output 변수를 딕셔너리 지정
#   
# =============================================================================

param = utils.get_config() #db_io_utils 의 get_config 함수 사용해서 config.ini 읽어오기(conrig 파일을 사이트별로 맞게 수정 하면 됨)

site = param['site'] #config 파일의 site를 site로 지정

raw_data_table = param['raw_data_table'] #config 파일의 raw_data_table을 raw_data_table로 지정
data_table = param['data_table']
pred_table = param['pred_table']

db_field_list = param['db_field_list']
db_field_type = param['db_field_type']

raw_field_dict = param['raw_field'] # config 파일의 raw_field를 raw_field_dict으로 지정

pred_field_dict = utils.make_pred_field_info(raw_field_dict, param['pred_field'], pred = True)  # olr 제외
pred_features_dict = utils.make_pred_field_info(raw_field_dict, param['pred_features'], pred=False)  # 결국 raw field dict과 같아지는데...


intp_field_dict = utils.make_field_info_suffix(raw_field_dict, '_intp') #raw_field_dict에 _intp를 붙임 ex. olr -> olr_intp
nw_field_dict = utils.make_field_info_suffix(raw_field_dict, '_nw')
nw_intp_field_dict = utils.make_field_info_suffix(raw_field_dict, '_nw_intp')

# update_field_list = intp_field_list + nw_field_list + nw_intp_field_list
update_field_dict = dict()
update_field_dict.update(intp_field_dict)
update_field_dict.update(nw_field_dict)
update_field_dict.update(nw_intp_field_dict)

# data_field_list = raw_field_list + update_field_list
data_field_dict = dict()
data_field_dict.update(raw_field_dict)
data_field_dict.update(update_field_dict)

normalize = 'standardize'

build = param['rebuild']

n_in = int(param['n_in'])
n_key = int(param['n_key'])
n_out = int(param['n_out'])

n_example = param['n_example']
fine_example = param['fine_example']
gap = param['gap']

batch_size = param['batch_size']

pre_epochs = 100
epochs = 10000
train_eps = 1e-5
fine_eps = 1e-5

conn_inf = [param['host'], param['user'], param['password'], param['database'], param['port']]
#config 파일의 host,user,password,database,port를 conn_inf로 지정

# ============================================================================





#=============================================================================
raw_date = '2019-09-30' # raw_date의 +5일뒤의 날짜부터 예측 시작 함.
#raw_date1 = '2020-08-31' #마지막 날짜
filename = './data_csv_last/ys_nw_data.csv' #여기있는 파일을 원데이터로 쓸 것.
test_period = 30
#=============================================================================




#DB 만들기 / table 만들기 / 데이터 보간하기 / 데이터 인서트 하기
#==============================================================================
'''
utils.create_database(conn_inf) #데이터 베이스 생성 

utils.make_code_data(conn_inf, './data_csv_last/item_id.csv') #아이템 아이디 테이블 생성


utils.create_table(conn_inf, raw_data_table, db_field_list, db_field_type)

#conn_inf,테이블 이름, 변수이름,변수타입 지정 -> "waterlaw_db" 테이블 생성

last_date = utils.insert_record(conn_inf, raw_data_table, filename, site, raw_field_dict, db_field_list, raw_date)
#DB에 데이터 인서트(raw_date 날짜 까지의 raw_data 인서트)

raw_data = utils.load_data(conn_inf, site, raw_data_table, raw_field_dict) #DB에 있는 데이터 불러오기
#raw_data2 = pd.read_csv(filename, header=None , index_col = None, names=['RegDate','olr', 'vfa', 'alk', 'vfa_alk', 'biogas_production', 'biogas_yield'])

linear_intp_data = utils.linear_interpolation(raw_data) #선형보간값
nw_data, nw_intp_data = utils.make_kde_data(raw_data, jack_start=0.1, jack_step=1, bin_start=2, bin_stop=3, bin_step=10) #nw 보간값
#nw_data : nw로 추정한 데이터(원데이터가 있어도 그것까지 추정값으로 대체됨 말그대로 추세예측한것.) 
#nw_intp_data : nw_data + 원데이터 (원데이터가 있는 부분은 원데이터로, 없는부분은 추정값으로 대체)
upload_data = np.concatenate([linear_intp_data, [data[1:] for data in nw_data], [data[1:] for data in nw_intp_data]], axis = 1)

utils.create_table(conn_inf, data_table, db_field_list, db_field_type) #optimal1 테이블생성
upload_data_ = utils.make_db_data(conn_inf, upload_data, site, update_field_dict)

utils.insert_(conn_inf, data_table, db_field_list, upload_data_) 
# optimal1에 업로드 -> linear_intp_data, nw_data, nw_intp_data 값이 DB에 업로드 되는것.
# nw_data는 나중에 dash에서 추세 그래프를 그릴때 필요하기 때문에 업로드 해야하는 것 같음.
##### insert_함수에서 start , end 떴으면 좋겠음
'''
#==============================================================================






#data의 날짜와 타임 웨잇을 계산하기 위한 과정.
#Time과 Time_weight을 계산해 주는 이유?
#else 이후의 작동에서 데이터를 업데이트 해야 하는데, 이때 날짜 간격을 계산해서 업데이트 해주기 위해.
#==============================================================================
from datetime import timedelta
raw_data2 = pd.read_csv(filename, header=None , index_col = None, 
                        names=['RegDate','olr', 'vfa', 'alk', 'vfa/alk' ,'biogas_production', 'biogas_yield'])

Data = raw_data2[raw_data2['RegDate']>=raw_date] #raw_date 이후의 날짜를 가져옴

    
#variables = Data.columns
start_time = datetime.datetime.strptime(str(Data['RegDate'].iloc[0]), '%Y-%m-%d') 
#strptime -> 다양한 포멧의 문자열을 datetime 객체로 변환

Time = list()
time_weight = list()

time_weight.append(0) # 각 날짜별로 얼마씩 차이 나는지 ex. 1.1, 1.3 -> time_weigth은 2임    
Time.append(0) # 몇일째 인지 확인 ex. 1.1,1.3,1.4,1.6-> 5일 
               # else 부분에서 다음 데이터를 업데이트 할 때 Time(==interval list)만큼 더해주기 위해 계산.
               # ex) 1.1, 1.5 -> Time 은 4가됨. 즉 1.1 + 4 == 1.5일의 데이터를 업데이트 하기위해.
               # ex2) 1.1, 1.5, 1.6 -> Time list는 [4,5]가됨. 즉 1.1 + 4 = 1.5일의 데이터 업데이트 하고, 그 다음 리스트 == 5일 즉 1.1+5 = 1.6일을 업데이트 하는 식.


for i in range(len(Data)-1):
    p_time = datetime.datetime.strptime(str(Data['RegDate'].iloc[i]),'%Y-%m-%d')
    f_time = datetime.datetime.strptime(str(Data['RegDate'].iloc[i+1]), '%Y-%m-%d')
    interval = int((f_time - p_time).days)
    
    time_weight.append(interval)
    
    if len(Time)==1:
        Time.append(interval)
    else :
        Time.append(Time[i] + interval)
        #if Time[i]+interval > 30:
        #    test_period = i + 1
        #    break
            

interval_list = list()
interval_list = Time       




# 학습 ========================================================================

save = param['save'] + str(test_period) + '/' 

prediction_time = list()

for i in range(test_period):

    if build:     

        print('Start pre-training...')

        nw_intp_data = utils.load_data(conn_inf, site, data_table, nw_intp_field_dict)
        #DB의 date_table(optimal1_db) 에서 nw_intp_field_dict에 맞는 데이터 불러 옴.
        #linear_intp_data = utils.load_data(conn_inf, site, data_table, intp_field_dict)
        nw_intp_data.columns = nw_intp_field_dict.values()
        #linear_intp_data.columns = intp_field_dict.values()
        
        
        pf = pred_features_dict.values() 
        #pred_features_dict : 실제 예측할 변수 # ?근데 왜 time olr이 있는지 모르겠음? 그냥 필요한 변수 다 가져온 느낌
        pred_features_data = nw_intp_data[pf] #linear_intp_data[pf] 
        #nw_intp_data에서 pf(실제 예측할 변수? 필요한 변수?)만 가져온 듯.
        
        start_step = 0
        pre_learning_rate_schedule = Networks.Schedule(12, start_step, 8 * 50)
        fine_learning_rate_schedule = Networks.Schedule(12, start_step, 8 * 50)
        
        pre_optimizer = tf.optimizers.Adam(pre_learning_rate_schedule, beta_1 = 0.9, beta_2 = 0.99, epsilon = 1e-9)
        fine_optimizer = tf.optimizers.Adam(fine_learning_rate_schedule, beta_1 = 0.9, beta_2 = 0.99, epsilon = 1e-9)
        
        print('Generating training data...', end = '')
        
        #generate_data 함수는 train 데이터, fine데이터 만들려고 하는것
        
        train_set, _ = Utils.generate_data(pred_features_data, n_in, n_key, n_out, n_example, gap, normalize, impute = False)
        #pred_features_data를 표준화한 결과 받음. (_는 로케이션,스케일값이라 안받은거 같음)
        #(n_example,input차원) ,(n_example, key차원) , (n_example, output차원) 이렇게 3덩어리로 나옴.
        print('Generating fine tunning data...', end = '')
        fine_set, stats = Utils.generate_data(pred_features_data, n_in, n_key, n_out, fine_example, gap, normalize, impute = False)
        #train_set과 같은 작업. 다만 쉐잎은 (fine_example,input차원) ,(fine_example, key차원) , (fine_example, output차원)
        print('Done.')
        
        num_train = len(train_set[0]) #n_example 개
        num_fine = len(fine_set[0]) #fine_example 개
        
        train_TAKE = (num_train // batch_size) + 1
        fine_TAKE = (num_fine // fine_example) + 1 

    
        
        train_data = tf.data.Dataset.from_tensor_slices(train_set) 
        #train_set의 첫번째 차원을 따라 슬라이스
        train_data = train_data.shuffle(buffer_size = num_train, reshuffle_each_iteration = True).repeat().batch(batch_size)
        # slices한 데이터를 shuffle해서 batch만큼씩 묶는데 그걸 repeat(무한번) 한다.
        
        fine_data = tf.data.Dataset.from_tensor_slices(fine_set)
        fine_data = fine_data.shuffle(buffer_size = num_fine, reshuffle_each_iteration = True).repeat().batch(batch_size)
        
        pred_input, pred_key = Utils.generate_data_for_predict(pred_features_data, stats, n_in, n_key)
        #제일 마지막 data에 대한 input , key 변수 표준화.
        #(generate_data_for_predict는 예측할때 쓸 데이터 만드는것.)
        
        model = Networks.Model_(n_out, pre_optimizer, fine_optimizer) #모델쌓기
        model.pre_train(train_data, batch_size, pre_epochs, train_TAKE, 1E-4) #프리트레이닝
        #fnn -> gbrbm ->bbrbm
        
        start_time = time.time()
        
        model.fine_tuning(fine_data, epochs, fine_TAKE, save, fine_eps) # 파인튜닝
        pred_key, pred_out, pred_input_key= model(pred_input, pred_key) 
        #표준화시킨 인풋과 키를 모델에 넣으면 key(input->key예측), out(key->out예측), out(input->key예측->out 예측) 예측      
        prediction_time.append(time.time() - start_time) #예측시간 계산
        print('Prediction time for %s took %.4fs' % (raw_date, time.time() - start_time))

        #DB업로드 작업==========================================================
        utils.create_table(conn_inf, pred_table, db_field_list, db_field_type) #pred_table(== optimal2_db) 테이블 생성
        
        #olr = Data.iloc[i , 1]

        #vsadd = olr * 13200

        utils.pred_db_upload(conn_inf, stats, pred_table, 
                             pred_field_dict , pred_key, pred_out, raw_date, gap) 
        #어...pred_input_key는 업로드 안하네? 그러네.. 그냥 input->key 예측한 pred_key와 key->output 예측한 pred_out만 업로드
        
        restore = save
        
        build = False
        #======================================================================
    else:
        test_date_ = datetime.datetime.strptime(raw_date, "%Y-%m-%d").date()
        
        test_date = test_date_ + datetime.timedelta(days = interval_list[i])
        filename = './data_csv_last/' + str(test_date) + '.csv'
        
        print('Start update waterlaw_db for %s...' % test_date, end = '')
        check = utils.insert_record(conn_inf, raw_data_table, filename, site, raw_field_dict, db_field_list, str(test_date))
        #raw_data_table('waterlaw_db') 테이블에 filename에 해당하는 데이터 업로드 (여기부터는 단일로 들어감 하루치 씩 들어간다고 생각하면 됨.)
        print('Done.')

        #insert 해서 업데이트된 새로운 raw_data load 해옴.    
        raw_data = utils.load_data(conn_inf, site, raw_data_table, raw_field_dict)
        #데이트 된 raw_data에서 비어있는 날짜 있으면 보간.======================
        linear_intp_data = utils.linear_interpolation(raw_data) 
        nw_data, nw_intp_data = utils.make_kde_data(raw_data,jack_start=0.1, jack_step=10, bin_start=10, bin_stop=51, bin_step=10)
        # nw_data, nw_intp_data = utils.make_kde_data(raw_data,jack_start=1, jack_step=3, bin_start=10, bin_stop=12, bin_step=1)
        #=====================================================================
        #DB에 올리기 위해 data 전처리==========================================
        upload_data = np.concatenate([linear_intp_data[-time_weight[i]:], [data[1:] for data in nw_data[-time_weight[i]:]], [data[1:] for data in nw_intp_data[-time_weight[i]:]]], axis = 1)
        upload_data_ = utils.make_db_data(conn_inf, upload_data, site, update_field_dict)
        #======================================================================
        print('Start update optimal1_db for %s...' % test_date, end = '')
        utils.insert_(conn_inf, data_table, db_field_list, upload_data_) #DB에 인서트
        print('Done.')

        #DB에 올라간 data를 load해옴.
        #linear_intp_data = utils.load_data(conn_inf, site, data_table, intp_field_dict)
        #linear_intp_data.columns = intp_field_dict.values()
        nw_intp_data = utils.load_data(conn_inf, site, data_table, intp_field_dict)
        nw_intp_data.columns = intp_field_dict.values()
 
        pf = pred_features_dict.values()
        pred_features_data = nw_intp_data[pf]
        #pred_features_data = linear_intp_data[pf]
        
        start_step = 0
        pre_learning_rate_schedule = Networks.Schedule(12, start_step, 8 * 50)
        fine_learning_rate_schedule = Networks.Schedule(12, start_step, 8 * 50)

        pre_optimizer = tf.optimizers.Adam(pre_learning_rate_schedule, beta_1 = 0.9, beta_2 = 0.99, epsilon = 1e-9)
        fine_optimizer = tf.optimizers.Adam(fine_learning_rate_schedule, beta_1 = 0.9, beta_2 = 0.99, epsilon = 1e-9)

        #여기서는 train_set 필요없음. (이미 위에서 pre_train 시킨 가중치 가지고 있음.)
        print('Generating fine tunning data...', end = '')
        fine_set, stats = Utils.generate_data(pred_features_data, n_in, n_key, n_out, fine_example, gap, normalize, impute = False)
        print('Done.')

        num_fine = len(fine_set[0])

        fine_TAKE = (num_fine // fine_example) + 1 
          
        fine_data = tf.data.Dataset.from_tensor_slices(fine_set)
        fine_data = fine_data.shuffle(buffer_size = num_fine, reshuffle_each_iteration = True).repeat().batch(batch_size)
        
        
        pred_input, pred_key = Utils.generate_data_for_predict(pred_features_data, stats, n_in, n_key)
        
        model = Networks.Model_(n_out, pre_optimizer, fine_optimizer)
        
        print('Load weights...', end = '')
        model.load_weights(restore)
        print('Done.')
        
        start_time = time.time()
        
        model.fine_tuning(fine_data, epochs, fine_TAKE, save, fine_eps)    ####################################
        pred_key, pred_out, pred_input_key = model(pred_input, pred_key)
    
        print('Prediction time for %s took %.4fs' % (test_date, time.time() - start_time))
        prediction_time.append(time.time() - start_time)        
 

        #olr = Data.iloc[i , 1]
        #vsadd = olr * 13200

        utils.pred_db_upload(conn_inf, stats, pred_table, pred_field_dict, 
                             pred_key, pred_out, test_date, gap)

        
        mape = utils.make_mape(conn_inf, raw_data_table, pred_table, raw_field_dict, pred_field_dict)
        mape_ = mape[0] # 아 mape 찍으면 세로로 나오는데 이걸 가로로 늘여뜨리려고 [0] 한거 같음!
        #MAPE.append(mape_)
        print(mape_)
        
 
mape_mean = np.mean(mape[0][1:])
print(mape_mean)
  
       
        
        
        
'''
ys_intp_data, ys_nw_data, ys_raw_data, variables, raw_field_dict = pickle.load(open('ys_data_.tlp', 'rb'))
    
start_time = datetime.datetime.strptime(str(ys_nw_data['Time'][-30:].iloc[0]), '%Y-%m-%d %H:%M:%S')
Time = list()  
for i in range(( ys_raw_data['Time'] >= start_time).sum()):
            
    num_true = ( ys_raw_data['Time'] >= start_time)
    interval_raw =  ys_raw_data['Time'][-num_true.sum():]
    f_time = datetime.datetime.strptime(str(interval_raw.iloc[i]), '%Y-%m-%d %H:%M:%S')
    
    interval = abs(int((start_time - f_time).days))

    Time.append(interval)
    

MAPE_ = list()

for i in range(len(Time)):
    MAPE_.append( MAPE[Time[i]] )
    
    
   
vfa_ = list()
alk_ = list()
production_ = list()
for i in range(len(MAPE_)):
    vfa_.append(MAPE_[i][1])
    alk_.append(MAPE_[i][2])
    production_.append(MAPE_[i][3])
    

print('vfa',np.mean(vfa_))
print('alk',np.mean(alk_))
print('produciotn', np.mean(production_))
print('mean', np.mean( [np.mean(vfa_), np.mean(alk_), np.mean(production_)]))
'''
    


'''
ys_intp_data, ys_nw_data, ys_raw_data, variables, raw_field_dict = pickle.load(open('ys_data_.tlp', 'rb'))
df = ys_intp_data.iloc[329:]


kde_g = KDE.KDE(df)
jackknife_g = kde_g.b_jackknife(start = 0.1, step = 10) #26.50918061482902
num_bins_g = kde_g.num_bins(jackknife_g, start = 1, stop = 50, step = 10) #11
NW_data_g = kde_g.NW_predict_original(num_bins_g, jackknife_g) #num_bins, b_jackknife
nw_data_g, nw_intp_data_g = kde_g.NW_interpolation(NW_data_g)
df.columns
nw_data_g2 = pd.DataFrame(nw_data_g, columns = ['Time','OLR(kg VS/m3)', 'kg_VSadd', 'VFA', 'Alkalinity', 'VFA/Alk ratio', 
            'Biogas Production(m3 x 4)', 'Biogas Yield_Vsadd(Nm3/kg Vsadd)']) #나다라야 추세
nw_intp_data_g2 = pd.DataFrame(nw_intp_data_g, columns = ['Time','OLR(kg VS/m3)', 'kg_VSadd', 'VFA', 'Alkalinity', 'VFA/Alk ratio', 
            'Biogas Production(m3 x 4)', 'Biogas Yield_Vsadd(Nm3/kg Vsadd)']) #원데이터 + 나다라야 보간



features = ['OLR(kg VS/m3)', 'kg_VSadd', 'VFA', 'Alkalinity', 'VFA/Alk ratio', 'Biogas Production(m3 x 4)', 'Biogas Yield_Vsadd(Nm3/kg Vsadd)']

#nw_data_g2 = nw_data_g2.iloc[329:]
#nw_intp_data_g2 = nw_intp_data_g2.iloc[329:]



A = utils.load_data(conn_inf, site, pred_table, pred_field_dict)

#예측값 인덱스 설정
g_index = list()
for i in range(249, 249 + test_period):
    g_index.append(i)

pprod_vfa = A.vfa_pred
pprod_alk = A.alk_pred
pprod_production = A.biogas_production_pred

pprod_vfa.index = g_index
pprod_alk.index = g_index
#pprod_vfa_alk.index = g_index
pprod_production.index = g_index
#pprod_yield.index = g_index
#pprod_new_yield.index = g_index



######## vfa
plt.style.use('seaborn')
fig = plt.figure(figsize=(18, 10))
top = fig.add_subplot(5,10,(5,19))


top.set_xlabel('Time')
top.set_ylabel('VFA')
top.set_title('VFA')

sub1_Left = 249
sub1_Right = 278

#top
top.scatter(nw_intp_data_g2.index[:-6][sub1_Left:sub1_Right], nw_intp_data_g2[:-6][features[2]][sub1_Left:sub1_Right], 15 ,color = 'gray') #나다라야 보간값 산점도
top.plot(nw_intp_data_g2[:-6][features[2]][sub1_Left:sub1_Right], color = 'gray') #나다라야 보간값 선
top.plot(nw_data_g2[:-6][[features[2]]][sub1_Left:sub1_Right], color = 'dodgerblue')#나다라야 추세
top.scatter(pprod_vfa.index,pprod_vfa, 10 ,color = 'red') # 예측값 산점도
top.plot(pprod_vfa, color = 'red') #예측값 선

#bottom
bottom = fig.add_subplot(2,2,(3,4))
bottom.set_xlabel('Time' )
bottom.set_ylabel('VFA' )
bottom.set_title('VFA')
bottom.scatter(nw_intp_data_g2.index[:-6], nw_intp_data_g2[:-6][features[2]], 15 ,color = 'gray') #나다라야 보간값 산점도
bottom.plot(nw_intp_data_g2[:-6][features[2]], color = 'gray' , label = 'Interpolation') #나다라야 보간값 선
bottom.plot(nw_data_g2[:-6][[features[2]]], color = 'dodgerblue' , label = 'NW')#나다라야 추세
bottom.scatter(pprod_vfa.index,pprod_vfa, 10 ,color = 'red') # 예측값 산점도
bottom.plot(pprod_vfa, color = 'red', label = 'pred') #예측값 선
bottom.legend(loc= 'upper left', fontsize=14,frameon=True,shadow=True)


bottom.fill_between((sub1_Left,sub1_Right), 50, 310, facecolor='green', alpha=0.2)
con1 = patches.ConnectionPatch(xyA=(sub1_Left, 90), coordsA=top.transData, 
                       xyB=(sub1_Left, 310), coordsB=bottom.transData, color = 'green')
fig.add_artist(con1)

con2 = patches.ConnectionPatch(xyA=(sub1_Right, 90), coordsA=top.transData, 
                       xyB=(sub1_Right,310 ), coordsB=bottom.transData, color = 'green')

fig.add_artist(con2)





#######alk
plt.style.use('seaborn')
fig = plt.figure(figsize=(18, 10))
top = fig.add_subplot(5,10,(5,19))

top.set_xlabel('Time')
top.set_ylabel('Alkalinity')
top.set_title('Alkalinity')

sub1_Left = 249
sub1_Right = 278

#top
top.scatter(nw_intp_data_g2.index[:-6][sub1_Left:sub1_Right], nw_intp_data_g2[:-6][features[3]][sub1_Left:sub1_Right], 15 ,color = 'gray') #나다라야 보간값 산점도
top.plot(nw_intp_data_g2[:-6][features[3]][sub1_Left:sub1_Right], color = 'gray') #원본에 나다라야 보간값 선
top.plot(nw_data_g2[:-6][[features[3]]][sub1_Left:sub1_Right], color = 'dodgerblue' )#나다라야 추세
top.scatter(pprod_alk.index,pprod_alk, 10 ,color = 'red' ) # 예측값 산점도
top.plot(pprod_alk, color = 'red') #예측값 선

#bottom
bottom = fig.add_subplot(2,2,(3,4))
bottom.set_xlabel('Time' )
bottom.set_ylabel('Alkalinity' )
bottom.set_title('Alkalinity')
bottom.scatter(nw_intp_data_g2.index[:-6], nw_intp_data_g2[:-6][features[3]], 15 ,color = 'gray') #나다라야 보간값 산점도
bottom.plot(nw_intp_data_g2[:-6][features[3]], color = 'gray', label = 'Interpolation') #나다라야 보간값 선
bottom.plot(nw_data_g2[:-6][[features[3]]], color = 'dodgerblue', label = 'NW')#나다라야 추세
bottom.scatter(pprod_alk.index,pprod_alk, 10 ,color = 'red') # 예측값 산점도
bottom.plot(pprod_alk, color = 'red', label = 'pred') #예측값 선
bottom.set_label('alk')
bottom.legend(loc= 'upper left', fontsize=14,frameon=True,shadow=True)




bottom.fill_between((sub1_Left,sub1_Right), 4200, 5700, facecolor='green', alpha=0.2)
con1 = patches.ConnectionPatch(xyA=(sub1_Left, 4500), coordsA=top.transData, 
                       xyB=(sub1_Left, 5700), coordsB=bottom.transData, color = 'green')
fig.add_artist(con1)
con2 = patches.ConnectionPatch(xyA=(sub1_Right, 4500), coordsA=top.transData, 
                       xyB=(sub1_Right,5700 ), coordsB=bottom.transData, color = 'green')
fig.add_artist(con2)




######production
plt.style.use('seaborn')
fig = plt.figure(figsize=(18, 10))
top = fig.add_subplot(5,10,(5,19))

top.set_xlabel('Time')
top.set_ylabel('Biogas Production(m3 x 4)')
top.set_title('Biogas Production(m3 x 4)')

sub1_Left = 249
sub1_Right = 278

#top
top.scatter(nw_intp_data_g2.index[:-6][sub1_Left:sub1_Right], nw_intp_data_g2[:-6][features[5]][sub1_Left:sub1_Right], 15 ,color = 'gray') #나다라야 보간값 산점도
top.plot(nw_intp_data_g2[:-6][features[5]][sub1_Left:sub1_Right], color = 'gray') #나다라야 보간값 선
top.plot(nw_data_g2[:-6][[features[5]]][sub1_Left:sub1_Right], color = 'dodgerblue')#나다라야 추세
top.scatter(pprod_production.index, pprod_production, 10 ,color = 'red') # 예측값 산점도
top.plot(pprod_production, color = 'red') #예측값 선

#bottom
bottom = fig.add_subplot(2,2,(3,4))
bottom.set_xlabel('Time' )
bottom.set_ylabel('Biogas Production(m3 x 4)' )
bottom.set_title('Biogas Production(m3 x 4)')
bottom.scatter(nw_intp_data_g2.index[:-6], nw_intp_data_g2[:-6][features[5]], 15 ,color = 'gray') #나다라야 보간값 산점도
bottom.plot(nw_intp_data_g2[:-6][features[5]], color = 'gray', label = 'Interpolation') #나다라야 보간값 선
bottom.plot(nw_data_g2[:-6][[features[5]]], color = 'dodgerblue', label = 'NW')#나다라야 추세
bottom.scatter(pprod_production.index, pprod_production, 10 ,color = 'red') # 예측값 산점도
bottom.plot(pprod_production, color = 'red', label = 'pred') #예측값 선
bottom.legend(loc= 'upper left', fontsize=14,frameon=True,shadow=True)


bottom.fill_between((sub1_Left,sub1_Right), 2000, 4500, facecolor='green', alpha=0.2)
con1 = patches.ConnectionPatch(xyA=(sub1_Left, 2450), coordsA=top.transData, 
                       xyB=(sub1_Left, 4500), coordsB=bottom.transData, color = 'green')
fig.add_artist(con1)
con2 = patches.ConnectionPatch(xyA=(sub1_Right, 2450), coordsA=top.transData, 
                       xyB=(sub1_Right, 4500), coordsB=bottom.transData, color = 'green')
fig.add_artist(con2)


'''






'''
print('mean',np.mean(MAPE_))
#col_mean = pd.DataFrame(np.mean(mape_,axis=0), index=['vfa','alk','vfa/alk','production','new_yield'])
col_mean = pd.DataFrame(np.mean(MAPE_,axis=0), index=['vfa','alk','production'])
print(col_mean)
'''




	





