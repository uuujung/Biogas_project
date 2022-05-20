"""
tunning learner
"""
print('Start import tensorflow...', end = '')

import datetime
import numpy as np
import time    

import db_io_utils as utils
import Utils_online as Utils

import Networks as Networks

import tensorflow as tf

#tf.random.set_seed(2020)
print('Done.')




param = utils.get_config()

site = param['site']

raw_data_table = param['raw_data_table']
data_table = param['data_table']
pred_table = param['pred_table']

db_field_list = param['db_field_list']
db_field_type = param['db_field_type']

raw_field_dict = param['raw_field']
pred_field_dict = utils.make_pred_field_info(raw_field_dict, param['pred_field'], pred = True)  # time, olr 제외
pred_features_dict = utils.make_pred_field_info(raw_field_dict, param['pred_features'], pred=False)  # 실제 예측할 변수

intp_field_dict = utils.make_field_info_suffix(raw_field_dict, '_intp')
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


filename = 'yangsan_0625_start.csv'
#filename = './data_csv/mae_data.csv'

############################################################################################################
raw_date = '2019-09-30' 
test_period = 5
############################################################################################################

########### make_db
# conn information
conn_inf = [param['host'], param['user'], param['password'], param['database'], param['port']]

print('Start make waterlaw_db...', end = '')

'''
utils.create_database(conn_inf)

utils.make_code_data(conn_inf, './data/item_id.csv')

utils.create_table(conn_inf, raw_data_table, db_field_list, db_field_type)

last_date = utils.insert_record(conn_inf, raw_data_table, filename, site, raw_field_dict, db_field_list, raw_date)
print('Done.')


print('Start make optimal1_db...', end = '')
raw_data = utils.load_data(conn_inf, site, raw_data_table, raw_field_dict)

linear_intp_data = utils.linear_interpolation(raw_data)

nw_data, nw_intp_data = utils.make_kde_data(raw_data)
upload_data = np.concatenate([linear_intp_data, [data[1:] for data in nw_data], [data[1:] for data in nw_intp_data]], axis = 1)

#upload_data = np.concatenate([linear_intp_data, [data[1:] for data in linear_intp_data], [data[1:] for data in linear_intp_data]], axis = 1)

utils.create_table(conn_inf, data_table, db_field_list, db_field_type)
upload_data_ = utils.make_db_data(conn_inf, upload_data, site, update_field_dict)

utils.insert_(conn_inf, data_table, db_field_list, upload_data_)
print('Done.')
'''



pred_field_dict = utils.make_pred_field_info(raw_field_dict, param['pred_field'], pred = True)  # time, olr 제외
pred_features_dict = utils.make_pred_field_info(raw_field_dict, param['pred_features'], pred=False)  # 실제 예측할 변수



###
save = param['save'] + str(test_period) + '/'

prediction_time = list()

for i in range(test_period):
    
    if build:     

        print('Start pre-training...')

        linear_intp_data = utils.load_data(conn_inf, site, data_table, intp_field_dict)
#        linear_intp_data = utils.load_data(conn_inf, site, raw_data_table, raw_field_dict)
        linear_intp_data.columns = intp_field_dict.values()
        
        pf = pred_features_dict.values()
        pred_features_data = linear_intp_data[pf] #아직 vfa/alk 계산 no
        
        start_step = 0
        pre_learning_rate_schedule = Networks.Schedule(12, start_step, 8 * 50)
        fine_learning_rate_schedule = Networks.Schedule(12, start_step, 8 * 50)
        
        pre_optimizer = tf.optimizers.Adam(pre_learning_rate_schedule, beta_1 = 0.9, beta_2 = 0.99, epsilon = 1e-9)
        fine_optimizer = tf.optimizers.Adam(fine_learning_rate_schedule, beta_1 = 0.9, beta_2 = 0.99, epsilon = 1e-9)
        
        print('Generating training data...', end = '')
        train_set, _ = Utils.generate_data(pred_features_data, n_in, n_key, n_out, n_example, gap, normalize, impute = False)
        print('Done.')
        print('Generating fine tunning data...', end = '')
        fine_set, stats = Utils.generate_data(pred_features_data, n_in, n_key, n_out, fine_example, gap, normalize, impute = False)
        print('Done.')
        
        num_train = len(train_set[0])
        num_fine = len(fine_set[0])
        
        train_TAKE = (num_train // batch_size) + 1
        fine_TAKE = (num_fine // fine_example) + 1 
 

        train_data = tf.data.Dataset.from_tensor_slices(train_set)
        train_data = train_data.shuffle(buffer_size = num_train, reshuffle_each_iteration = True).repeat().batch(batch_size)
        
        fine_data = tf.data.Dataset.from_tensor_slices(fine_set)
        fine_data = fine_data.shuffle(buffer_size = num_fine, reshuffle_each_iteration = True).repeat().batch(batch_size)
        
        
        pred_input, pred_key = Utils.generate_data_for_predict(pred_features_data, stats, n_in, n_key)
        
        model = Networks.Model_(n_out, pre_optimizer, fine_optimizer)
        model.pre_train(train_data, batch_size, pre_epochs, train_TAKE, train_eps)
        
        start_time = time.time()
        
        model.fine_tuning(fine_data, epochs, fine_TAKE, save, fine_eps)    ####################################
        pred_key, pred_out, pred_input_key = model(pred_input, pred_key)
        
        prediction_time.append(time.time() - start_time)
        print('Prediction time for %s took %.4fs' % (raw_date, time.time() - start_time))

        utils.create_table(conn_inf, pred_table, db_field_list, db_field_type)
        
        utils.pred_db_upload(conn_inf, stats, pred_table, pred_field_dict, 
                             pred_key, pred_out, raw_date, gap)

        restore = save
        
        build = False
        
    else:
        
        test_date_ = datetime.datetime.strptime(raw_date, "%Y-%m-%d").date()
        
        test_date = test_date_ + datetime.timedelta(days = i)
        filename = './data/' + str(test_date) + '.csv'
        
        print('Start update waterlaw_db for %s...' % test_date, end = '')
        check = utils.insert_record(conn_inf, raw_data_table, filename, site, raw_field_dict, db_field_list, str(test_date))
        print('Done.')

        raw_data = utils.load_data(conn_inf, site, raw_data_table, raw_field_dict)

        linear_intp_data = utils.linear_interpolation(raw_data)

        nw_data, nw_intp_data = utils.make_kde_data(raw_data)
        upload_data = np.concatenate([linear_intp_data, [data[1:] for data in nw_data], [data[1:] for data in nw_intp_data]], axis = 1)
        #upload_data = np.concatenate([linear_intp_data, [data[1:] for data in linear_intp_data], [data[1:] for data in linear_intp_data]], axis = 1)
        upload_data_ = utils.make_db_data(conn_inf, upload_data, site, update_field_dict)

        print('Start update optimal1_db for %s...' % test_date, end = '')
        utils.insert_(conn_inf, data_table, db_field_list, upload_data_)
        print('Done.')

#        linear_intp_data = utils.load_data(conn_inf, site, raw_data_table, raw_field_dict)
#        linear_intp_data.columns = raw_field_dict.values()
        linear_intp_data = utils.load_data(conn_inf, site, data_table, intp_field_dict)
        linear_intp_data.columns = intp_field_dict.values()
 
        pf = pred_features_dict.values()
        pred_features_data = linear_intp_data[pf]
        
        start_step = 0
        pre_learning_rate_schedule = Networks.Schedule(12, start_step, 8 * 50)
        fine_learning_rate_schedule = Networks.Schedule(12, start_step, 8 * 50)

        pre_optimizer = tf.optimizers.Adam(pre_learning_rate_schedule, beta_1 = 0.9, beta_2 = 0.99, epsilon = 1e-9)
        fine_optimizer = tf.optimizers.Adam(fine_learning_rate_schedule, beta_1 = 0.9, beta_2 = 0.99, epsilon = 1e-9)

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

        utils.pred_db_upload(conn_inf, stats, pred_table, pred_field_dict, 
                             pred_key, pred_out, test_date, gap)

        mape = utils.make_mape(conn_inf, raw_data_table, pred_table, raw_field_dict, pred_field_dict)
        mape_ = mape[0]
        print(mape_)
        
mape_ = np.mean(mape[0][1:])
print(mape_)
 