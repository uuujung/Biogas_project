
import configparser
import json

import pandas as pd

import pymysql #mysql을 python에서 사용 할 수 있는 라이브러리

import datetime

import numpy as np

from kde import KDE as kde_


# config파일 읽어오는 함수
def get_config(config_filepath = 'config.ini'):
    #config에서 숫자는 int로, 키-밸류는 json을 사용해 딕셔너리로,
    #문자는 str으로, 경로는 config 그대로 불러오는것을 알 수 있음.
    
    config = configparser.RawConfigParser()
    config.read(config_filepath) #'config_intp.ini')
#    config.read('./config_intp.ini')
    
    param = dict()
    
    param['site'] = str(config['SOURCE']['SITE'])
    param['host'] = str(config['SOURCE']['HOST'])
    param['user'] = str(config['SOURCE']['USER'])
    param['password'] = str(config['SOURCE']['PASSWORD'])
    param['database'] = str(config['SOURCE']['DATABASE'])
    param['port'] = int(config['SOURCE']['PORT'])
    param['raw_data_table'] = str(config['SOURCE']['RAW_DATA_TABLE'])
    param['data_table'] = str(config['SOURCE']['DATA_TABLE'])
    param['pred_table'] = str(config['SOURCE']['PRED_TABLE'])
     
    param['vars'] = json.loads(config['VARS']['VARS']) 
    #json문자열을 python의 객체로 변환하기 위해 loads()를 쓴다?
    #json의 키밸류->파이썬의 딕셔너리로 변환 됨.  아 vars 부터는 키-밸류 관계여서
    #파이썬 딕셔너리로 변환하기 위해 json.loads쓰는듯
    param['raw_field'] = json.loads(config['VARS']['RAW_FIELD'])
    param['raw_code'] = json.loads(config['VARS']['RAW_CODE'])
    param['pred_field'] = json.loads(config['VARS']['PRED_FIELD'])
    param['pred_features'] = json.loads(config['VARS']['PRED_FEATURES'])
    param['db_field_list'] = json.loads(config['VARS']['DB_FIELD_LIST'])
    param['db_field_type'] = json.loads(config['VARS']['DB_FIELD_TYPE'])

    param['n_in'] = int(config['PARAMETERS']['N_IN'])
    param['n_key'] = int(config['PARAMETERS']['N_KEY'])
    param['n_out'] = int(config['PARAMETERS']['N_OUT'])
    
    param['n_example'] = int(config['PARAMETERS']['N_EXAMPLE'])
    param['fine_example'] = int(config['PARAMETERS']['FINE_EXAMPLE'])
    
    param['batch_size'] = int(config['PARAMETERS']['BATCH_SIZE'])
    param['gap'] = int(config['PARAMETERS']['GAP'])
    
    units = json.loads(config['TUNING_PARAMETERS']['units'])
    activations = json.loads(config['TUNING_PARAMETERS']['activations'])
    initializers = json.loads(config['TUNING_PARAMETERS']['initializers'])
    
    param['fnn_units'] = units['fnn'] #units가 키-밸류(딕셔너리)상태인데 거기서 fnn'값' 만을 가져옴
    param['fnn_activations'] = activations['fnn']
    param['fnn_initializers'] = initializers['fnn']
    
    param['rbm_units'] = units['rbm']
    param['rbm_activations'] = activations['rbm']
    param['rbm_initializers'] = initializers['rbm']
    
    param['rebuild'] = bool(config['BUILD']['REBUILD']) #bool : True of False로 뱉어냄
    
    param['gpu'] = int(config['GPU']['ID'])
    param['memory_growth'] = bool(config['GPU']['GROWTH'])
    
    param['save'] = config['MODEL']['SAVE']
    param['restore'] = config['MODEL']['RESTORE']

    return param        

#왜 pred_field를 만들때 raw_field랑 다른 방법을 사용하는지 모르겠음.
#config에서 처음부터 dict으로 받아서 get_config 수정보면 똑같은 방식으로 불러올수있는데
#굳이 config에서 list로 받아서 이런 과정을 거치는 이유는....??
#쨌든 make_pred_field_info의 역할은 pred 데이터의 필드 딕셔너리 생성.
def make_pred_field_info(raw_field, field_list, pred):
    
    fields = dict()
    
    if pred:
        for field in field_list:
            if field == 'RegDate':
                fields[field] = raw_field[field]
            else:
                fields[str(field + '_pred')] = raw_field[field]
    else:
        for field in field_list:
            fields[field] = raw_field[field]
    
    return fields


# suffix(접미사)에 따른 field dict만드는 함수
def make_field_info_suffix(raw_field, suffix):
    
    fields = dict()        
    
    for key, value in raw_field.items():
        if key == 'RegDate':
            fields[key] = value
        else:
            fields[key + suffix] = value
            
    return fields


# csv파일 load
def get_csv(filename, until):
    #################################################
    """field_dictionary"""
    
    data = pd.read_csv(filename, header = None)  
    data = data.values
    
    end_date = datetime.datetime.strptime(until, "%Y-%m-%d").date()
    
    data_ = list()
    for row in data:
        if not pd.isnull(row[0]):
            row_ = list()
            date = datetime.datetime.strptime(str(row[0]), "%Y-%m-%d").date()
            if date <= end_date:
                row_.append(date)
                for r in row[1:]:
                    row_.append(float(r)) 
                data_.append(row_)   
        
    return data_
     

# pymysql connection 부르기
def get_conn(host_, user_, password_, database_, port_):
    
    if database_ is not None:
        #pymysql.connect() 메소드 사용하여 mysql에 연결.
        conn = pymysql.connect(host = host_,
                           user = user_,
                           password = password_,
                           db = database_,
                           port = port_,
                           charset = 'utf8')    
    else:
        conn = pymysql.connect(host = host_,
                           user = user_,
                           password = password_,
                           port = port_,
                           charset = 'utf8')             
    
    return conn

# sql문 실행
def execute_sql(host, user, password, database, port, sql, data): 
    #DB 생성, table생성은 data가 필요없지만, insert 같은경우는 data 필요하기 때문에 data = None 인경우와 아닌경우를 나눠놓음
    
    conn = get_conn(host, user, password, database, port)

    if data is None:
        #연결한 DB와 상호작용하기 위해 cursor 객체 생성.
        with conn.cursor() as cursor:
            cursor.execute(sql) #cursor 객체의 execute()메서드를 사용해 문장을 DB 서버에 보낸다
        
        conn.commit() #commit 한다 (DML 언어를 사용 할 때는 항상 커밋해줘야 함)
        cursor.close() #DB연결을 닫는다
        
    else:
        with conn.cursor() as cursor:
            cursor.execute(sql, data)
        conn.commit()
        cursor.close()


# sql과 statement를 띄어쓰기로 합치기
# 아이걸 쓰는 곳에 넣으면 안되는 이유가 다방면에 걸쳐서 사용하네!
def generate_sql(sql, statement):
    return sql + ' ' + statement


# database drop sql문과 create sql문 생성
def create_database_sql(host, user, password, database, port):
    
    drop_sql = 'DROP DATABASE IF EXISTS'
    drop_sql = generate_sql(drop_sql, database) #문자임
    
    create_sql = 'CREATE DATABASE'
    create_sql = generate_sql(create_sql, database) #이건 문자일 뿐이야

    return drop_sql, create_sql


# database 만드는 함수
def create_database(conn_inf):
    
    host, user, password, database, port = conn_inf
    
    drop_database_sql_, create_database_sql_ = create_database_sql(host, user, password, database, port)
    execute_sql(host, 'root', password, None, port, drop_database_sql_, data = None) #excute_sql 함수를 이용해 sql문을 실행.
    execute_sql(host, 'root', password, None, port, create_database_sql_, data = None)
    #execute_sql(host, user, password, database, port, sql, data)
      

#아이템 아이디 테이블 및 데이터 인서트 
def make_code_data(conn_inf, data_path):
    
    table_name = 'item_id'
    field_list = ['item_nm', 'item_cd']
    field_type = ['VARCHAR(50)', 'VARCHAR(10)']
    
    create_table(conn_inf, table_name, field_list, field_type)
    
    host, user, password, database, port = conn_inf
    
    item_id = pd.read_csv(data_path, header = None)
    
    item_id_ = item_id.values
    
    item_list = list()
    for item in item_id_:
        tmp = list()
        for i, it  in enumerate(item):
            if i < 1:
                tmp.append(it)
            else:
                it = it.replace("'", "")
                tmp.append(it)
        item_list.append(tmp)
    
    # insert문이 기존과 다르게 들어가야 해서 임시로 이렇게 설정해놓음
    insert_table_sql = 'INSERT INTO item_id (item_nm, item_cd) VALUES (%s, %s)'  
    for d_ in item_list:  
        execute_sql(host, user, password, database, int(port), insert_table_sql, d_)

  

# table drop sql문과 create sql문 생성
def create_table_sql(table_name, field_list , field_type):
    #테이블을 생성 할때는 열 특성을 정해줘야하기 때문에
    #field_list는 열이름, field_type은 데이터 타입 지정을 위해 필요.

    sql = 'CREATE TABLE IF NOT EXISTS'
    sql = generate_sql(sql,table_name)
    sql = generate_sql(sql, '(')
    
    for i, field in enumerate(field_list):
        if field == field_list[-1]:
            sql += str(field)
            type_ = str(field_type[i])
            sql = generate_sql(sql, type_ + ')')
            sql = generate_sql(sql, 'ENGINE=InnoDB')
            sql = generate_sql(sql, 'DEFAULT')
            sql = generate_sql(sql, 'CHARSET=utf8')     
            
        else:
            sql += str(field)
            type_ = str(field_type[i])
            sql = generate_sql(sql, type_ + ',')
            
    return sql

    
# table 만드는 함수
def create_table(conn_inf, table_name, field_list, field_type):
    
    host, user, password, database, port = conn_inf
    
    drop_database_sql = 'DROP TABLE IF EXISTS' + ' ' + str(table_name)
    execute_sql(host, user, password, database, port, drop_database_sql, data = None)
    
    create_table_sql_ = create_table_sql(table_name, field_list, field_type) #sql문 생성
    execute_sql(host, user, password, database, port, create_table_sql_, data = None)
    
    
def get_code(conn_inf, field_list):
    #여기서의 field_list는 아래 make_db_data에서 field_dict.key() 가 된다.

    host, user, password, database, port = conn_inf
    conn = get_conn(host, user, password, database, port)
    
    sql = select_sql('item_id', None)
    
    cur = conn.cursor()
    cur.execute(sql) #item_id테이블에 있는 모든 데이터를 셀렉트하라는 명령문을 DB에 보냄
    rows = cur.fetchall() #아마 DB에서 데이터 불러오는거 같은데
    cur.close()
    
    df = pd.DataFrame(rows) #item_id 데이터 나옴
    
    df.columns = ['item_nm', 'item_cd'] #아이템 이름, 아이템 코드
    
    code_list = list()
    for field in field_list:
        if field == 'RegDate':
            pass
        else:
            #item_id의 이름 중에 field(ex.'RegDate', 'olr_intp', 'vfa_intp' 같은 것들)와 
            #동일한 것의 아이템 코드('item_cd')를 code_list에 모으겠다.
            code_list.append(df[df['item_nm'] == field]['item_cd'].item())
        
    return code_list #item id의 코드만 가져와짐
 
# 데이터를 테이블의 형태에 맞게 바꿔주는 함수
def make_db_data(conn_inf, data, site, field_dict):
 
    param = get_config()
    
    #(if)우리가 DB에 업로드 해야할 가공된 데이가 raw_data와 동일한 변수를 가지고 있다면 field_list는 그냥 raw_data의것을 그대로 쓰면 될것이고
    #(else)그렇지 않다면 새로받은 field_dict의 keys를 field_list로 사용하자. 
    if field_dict.keys() == param['raw_code'].keys():
        field_list = param['raw_code'].values()  
    else:
        field_list = field_dict.keys()
        
    field_code = get_code(conn_inf, field_list)
    
    field_code = ['RegDate'] + field_code
    
    data = pd.DataFrame(data, columns = field_code) # 여기서 열이 아이템 아이디라고 보면됨.
    
    #melt == 데이터 재 구조화
    #id_vars == 기준이 될 cloumn / value_vars == 세로로 변형될 data
    #즉 RegDate 기준으로 item id 코드가 세로로 정렬된다고 보면됨.
    #데이터 형태는 노션 참고
    data = pd.melt(data, id_vars = 'RegDate', value_vars = list(data.columns[1:]), #[1:]해주는 이유 : 맨 앞에 RegDate있어서 제외시키려고 
                     var_name = 'itemID_cd', value_name = 'value')
    
    #melt된 데이터를 날짜별로 sort함 이때 reset_index 통해서 인덱스 리셋시키고 0~새로운 인덱스 부여
    data = data.sort_values(by = 'RegDate').reset_index(drop = True)
    
    data['site_id'] = site #site_id 라는 열 추가 후 site를 값으로
    
    data = data[['site_id', 'itemID_cd', 'RegDate', 'value', 'itemID_cd', 'RegDate']] #[]순서대로 열 정렬
    
    data['RegDate'] = data['RegDate'].astype(str)
    
    data = data.values
    
    data_list = list()
    for d in data:
        data_list.append([di for di in d]) 
        # 이때 그냥 append(d)하면 array까지 같이 어펜드 됨. for 두번써서 array빼고 값만 어펜드 하는데 원리는 모르겠음.
        
    return data_list
    

# insert table sql문 생성
# 똑같은 itemID_cd와 RegDate를 갖는 데이터가 이미 존재하면 insert하지 않고 존재하지 않으면 insert함
def insert_table_sql(table_name, field_list):
    
    sql = 'INSERT INTO'
    sql = generate_sql(sql, table_name)
    sql = generate_sql(sql, '(')
    
    values = "" #인서트할 값을 채워나가야 함 이제
    
    for i, field in enumerate(field_list):
        
        if i < len(field_list) - 1:
            sql += str(field) + ','
            values += '%s,'
        else:
            sql += str(field) + ')'
            values += '%s'
            sql = generate_sql(sql, 'SELECT')
            sql = generate_sql(sql, str(values))
            
    sql = generate_sql(sql, 'FROM dual WHERE NOT EXISTS (SELECT * FROM') 
    #dual은 계산을위한 임시 테이블? 정도로 생각하면됨 -> 노션 링크 참고
    #select 함수를 통해 조건(where)을 충족하는 데이터만 인서트 하고 싶기 때문에 이때
    sql = generate_sql(sql, table_name)
    sql = generate_sql(sql, 'WHERE itemID_cd = %s AND RegDate = %s)')
        
    return sql 
#'INSERT INTO new_tset (site_id,itemID_cd,RegDate,value) SELECT %s,%s,%s,%s FROM dual WHERE NOT EXISTS (SELECT * FROM new_tset WHERE itemID_cd = %s AND RegDate = %s)'
# SELECT %s,%s,%s,%s <- 얘네가 인서트 할 값들인데 얘가 형성되어있는 table이 없으니 dual 테이블으로 설정한것.
# WHERE부터는 조건(=똑같은 itemID_cd와 RegDate를 갖는 데이터가 이미 존재하면 insert하지 않고 존재하지 않으면 insert함)
# %s가 먹나보네..?


# data를 table에 insert하는 함수
def insert_(conn_inf, table_name, field_list, data_):
    
    host, user, password, database, port = conn_inf
    
    insert_table_sql_ = insert_table_sql(table_name, field_list)
    #print('insert satrt...')
    for data in data_: #for 돌려서 data의 행 한줄 한줄을 인서트하는거네 -> 좀더 효율적인 방법은 DB공부를 해봐야 알 수 있을듯.
        execute_sql(host, user, password, database, port, insert_table_sql_, data) 
    #print('...insert Done')
    
    
#애는 csv파일을 인서트 하는거고 위에꺼는 python에서 가공한 데이터를 인서트 하는거 같음.
def insert_record(conn_inf, table, filename, site, raw_field_dict, db_field_list, date):
    
    host, user, password, database, port = conn_inf
 
    data = get_csv(filename, date)
    
    data_ = make_db_data(conn_inf, data, site, raw_field_dict)
    
    insert_(conn_inf, table, db_field_list, data_)

    return get_last_RegDate(conn_inf, table)
    
    
# select sql문 생성
# SELECT field_list FROM table_name
def select_sql(table_name, field_list):

    sql = 'SELECT'
    
    if field_list is not None:
        for i, field in enumerate(field_list):
            if i < (len(field_list) - 1):
                sql = generate_sql(sql, field)
                sql += ',' 
            else:
                sql = generate_sql(sql, field)
    else:
        sql = generate_sql(sql, '*')
            
    sql = generate_sql(sql, 'FROM')
    sql = generate_sql(sql, str(table_name))
    
    return sql

############################################??? 어디에쓰는거지 -> 현재는 안씀
def get_last_RegDate(conn_inf, table):
    
    host, user, password, database, port = conn_inf    
    conn = get_conn(host, user, password, database, port)
    
    cursor = conn.cursor()
    cursor.execute('select max(RegDate) from waterlaw_db')
    rows = cursor.fetchone()
    cursor.close()

    return rows[0].date()
###############################################


# 데이터베이스에서 데이터 불러오는 함수        
def load_data(conn_inf, site, table_name, field_dict):
    
    host, user, password, database, port = conn_inf
    param = get_config()
    
    if field_dict.keys() == param['raw_code'].keys():
        field_list = param['raw_code'].values()
    else:
        field_list = field_dict.keys()

    field_code = get_code(conn_inf, field_list) # field_list에 해당하는 아이템아이디코드
    
    conn = get_conn(host, user, password, database, port)
    
    sql = select_sql(table_name, None) #SELECT sql문 생성
    
    cur = conn.cursor()
    cur.execute(sql)
    rows = cur.fetchall() #데이터 불러오기
    cur.close()
    
    data = pd.DataFrame(rows)
    
    data_field = ['site_id', 'itemID_cd', 'RegDate', 'value']
    data.columns = data_field
    
    data = data.loc[data['site_id'] == site, :] #data.loc[행조건 , 열조건] 
    
    data = data.drop(['site_id'], axis = 1)
    
    #data 재배열 해줄것. 이때 index(행) = RegDate, columns(열) = 'itemID_cd' 를 기준으로 'value' 데이터를.
    data = data.pivot(index = 'RegDate', columns = 'itemID_cd')['value'].reset_index()
        #= data.pivot(index = 'RegDate', columns = 'itemID_cd', values='value').reset_index()

    #여기까지의 data의 열 순서는 알파벳순인가 여튼 우리가 정한 순서는 아님. 그걸 우리 필드 코드 순서랑 동일하게 맞춰주기 위해 아래 코드 시행하는 것.
    field_code = ['RegDate'] + field_code #두개를 합한 리스트 형태.
    data = data[field_code] #field_code의 순서대로 데이터 재정렬됨.
    
    data.columns = field_dict.keys() #itemID가 변수별로 나눠논거니까 load할 때는 알아보기 쉽게 변수명으로 열 이름을 바꿔주기 위해 시행.
    
    return data


# 선형보간 함수 >> 1222 itv2 추가, range 범위 수정
def linear_interpolation(data):

    import datetime
    from datetime import timedelta

    variables = data.columns
    
    start_time = datetime.datetime.strptime(str(data['RegDate'].iloc[0]), '%Y-%m-%d %H:%M:%S')
    #print(start_time)
    #end_time = datetime.datetime.strptime(str(data[-1][0]), "%Y-%m-%d")

    new_data = list()
        
    for i in range(len(data) - 1):
        
        #new_data.append(data['Time'][i])
        
        p_time = datetime.datetime.strptime(str(data['RegDate'].iloc[i]), '%Y-%m-%d %H:%M:%S')
        f_time = datetime.datetime.strptime(str(data['RegDate'].iloc[i+1]), '%Y-%m-%d %H:%M:%S')
        interval = int((f_time - p_time).days)
        
        if interval == 1 :
            new_data.append([data['RegDate'][i]] + [d for d in data.loc[i].values[1:]])
            #pass
        else:
            x1 = int((p_time - start_time).days)#int(data[i][0])
            x2 = int((f_time - start_time).days)#int(data[i + 1][0]
            
            for itv in range(0, interval): #여기 range(1, interval) -> (0, interval)로 수정
                tmp = list()
                x = x1 + itv
                itv2 = p_time + timedelta(days=itv) #여기 추가함 
                x_time = str(datetime.datetime.strftime(itv2, "%Y-%m-%d"))
                tmp.append(x_time)
                
                for var in variables[1:]:
                    
                    y1 = float(data[var].loc[i])
                    y2 = float(data[var].loc[i + 1])
                    m = (y2 - y1)/(x2 - x1)
                    q = y1 - m * x1
                    y = m * x + q
                    tmp.append(y)
                    
                new_data.append(tmp)
                
    new_data.append([data['RegDate'].iloc[-1]] + [d for d in data.iloc[-1][1:]])
                    
    new_data_ = list()
    for row in new_data:
        tmp = list()
        tmp.append(str(row[0]))
        for r in row[1:]:
            tmp.append(np.float(r))
        #print(tmp)
        new_data_.append(tmp)
                    
    return new_data_  


# kde 데이터 만드는 함
def make_kde_data(data, jack_start, jack_step, bin_start, bin_stop, bin_step):
    
    kde = kde_(data)
    
    jackknife = kde.b_jackknife(jack_start, jack_step) #ked.b_jackknife(0.1,10) -> 인수 받을 수 있도록 수정 함
    n_bins = kde.num_bins(jackknife, bin_start, bin_stop, bin_step)  #원래는 (jackknife, 10, 301, 10) > (10,81,10)-> 인수 받을 수 있도록 수정 함
    NW_data = kde.NW_predict(n_bins, jackknife)
    nw_data, intp_data = kde.NW_interpolation(NW_data)

    nw_data_ = list()
    for row in nw_data:
        tmp = list()
        tmp.append(str(row[0]))
        for r in row[1:]:
            tmp.append(np.float(r))
        nw_data_.append(tmp)

    intp_data_ = list()
    for row in intp_data:
        tmp = list()
        tmp.append(str(row[0]))
        for r in row[1:]:
            tmp.append(np.float(r))
        intp_data_.append(tmp)
        
    return nw_data_, intp_data_


# 예측값 테이블에 넣는 함수     
def pred_db_upload(conn_inf, stats, pred_table, pred_field, 
                   pred_key, pred_out, time, gap): 
    #time은 raw data길이(raw_date로 쓰네..?) #pred_field는 예측해야될 변수 

    host, user, password, database, port = conn_inf
    param = get_config()

    #import datetime
    
	# pred_key , pred_out 정규화 =========================================================
    location, scale = stats #평균, 표준편차
    #stats가 크게 2덩어리로 나뉨 -> 큰 덩아리 하나가 또 3덩어리로 나뉨
    location_X, location_Y1, location_Y2 = location 
    scale_X, scale_Y1, scale_Y2 = scale

    key_pred = np.squeeze((pred_key * scale_Y1) + location_Y1) # (1,3)을 (3,)로 쉐잎변경
    #표준화 되어있던 값을 다시 원래대로 돌리는 작업.
    #key_pred_ = key_pred[0] / key_pred[1] #내 기억으로 이거 vfa/alk 때문에 있었던 계산임.
    #key_pred = np.append(key_pred, key_pred_)
    
    out_pred = np.squeeze((pred_out * scale_Y2) + location_Y2) # (1,3)을 (3,)로 쉐잎변경
    #out_pred_ = out_pred / vsadd #일드 계산때문에 있었던 식임. 없어도됨
    #out_pred = np.append(out_pred, out_pred_)
	#=======================================================================================    

    pred_data = list()
    
	#예측값의 날짜를 계산하기 위한 부분. =====================================================
    raw_time = datetime.datetime.strptime(str(time), '%Y-%m-%d')#' %H:%M:%S') 
	#raw_date의 형태를 '%Y-%m-%d' 로 바꿔줌
    pred_time = raw_time + datetime.timedelta(days = gap) 
    # timedelta : 시간 연산을 위해 사용 -> raw_time(오늘) + timedelt(gap=5) = 오늘+5일 후 를 뜻함 
    pred_time = datetime.datetime.strftime(pred_time, '%Y-%m-%d')#' %H:%M:%S')
    
	# 참고 strftime / strptime 차이점
	# strftime : 객체 -> 문자열 , strptime : 문자열 -> 객체
    # 따라서 계산 필요할 때는 객체로 변경하는 ptime을 쓰고 다시 문자로 변경이 필요할 때 ftime 씀.
    #=========================================================================================
    pred_data.append(pred_time)
    

	# key_pred(key 예측값)을 차례대로 pred_data라는 list에 담음 ================================
    for di in key_pred:
        pred_data.append(np.float(di)) #[오늘+gap의 날짜,키1,키2,키3]값이 담긴 list
    
    #pred_data.append(np.float(out_pred)) #아웃풋이 하나만 있을때(일드 없을 때)
    
	# out_pred(out 예측값)을 차례대로 pred_data 라는 list에 담음 ==============================
    for di in out_pred:
        pred_data.append(np.float(di)) # [오늘+gap의 날짜,키1,키2,키3,아웃풋1,아웃풋2] 값이 담긴 list
    #print(pred_data)
    #===========================================================================================

    pred_data_ = list()
		#[[오늘+gap의 날짜,키1,키2,키3,아웃풋1,아웃풋2]] *list 안에 리스트 형태 됨. ==================
    pred_data_.append(pred_data) 
    

    #make_db_data로 pred_data를 DB의 형태에 맞게 변경시켜줌.
    pred_data_ = make_db_data(conn_inf, pred_data_, param['site'], pred_field)


    insert_sql = insert_table_sql(pred_table, param['db_field_list'])
    #'INSERT INTO optimal2_db (site_id,itemID_cd,RegDate,value) SELECT %s,%s,%s,%s FROM dual WHERE NOT EXISTS (SELECT * FROM optimal2_db WHERE itemID_cd = %s AND RegDate = %s)'
    #라는 sql문을 생성.

    #execute 함수를 통해 만들어 놓은 insert sql문 실행해서 디비에 인서트함.
    for pred in pred_data_:
        execute_sql(host, user, password, database, port, insert_sql, data = pred)
    





# mape계산해주는 함수  
def make_mape(conn_inf, raw_data_table, pred_table, raw_field_dict, pred_field_dict):
    
    host, user, password, database, port = conn_inf
    mape_field = [k for k in pred_field_dict.keys()] #mape_필드니까 key로 받는거 같음. 필드가 밸류는 아니니까

    
    param = get_config()
    n_in = param['n_in'] #input 갯수 
    site = param['site']
    
	# DB에 업로드 되어있는 raw_data와 pred_data 로드 받기 ===============================
	# 학습시 매턴 마다 raw_data와 pred_date가 DB에 하루씩업로드 됨. 즉 매 턴마다 새로 업데이트된 데이터를 받아서 mape 계산할 것.
    raw_data = load_data(conn_inf, site, raw_data_table, raw_field_dict) 
    pred_data = load_data(conn_inf, site, pred_table, pred_field_dict)
    
    raw_data = raw_data.values
    pred_data = pred_data.values
	# ==================================================================================
    
	# raw_data와 pred_data가 같은 날짜일 때 각각 raw_d와 pred_d에 어펜드 하려고 함 ========
	# 같은 날짜일때 어펜드 하는 이유? -> 원본과 예측값에 대한 mape를 구하기 위해!
    raw_d = list()
    pred_d = list()
    for pred in pred_data:
        for raw in raw_data:
            if pred[0] == raw[0]: #예측이랑 원본이랑 날짜 같으면 (두개 비교하려고)
                raw_d.append(raw[(n_in + 1):]) #raw의 1번째는 olr이라서 제외 시키는것. 
                pred_d.append(pred[1:])

    #if len(raw_d) > 0:
    #    raw_d = np.asarray(raw_d[0][1:])

    raw_d = np.asarray(raw_d)
    pred_d = np.asarray(pred_d)
	#=======================================================================================
  

    # mape 공식에 따라 함수 구현 ============================================================
    def mape(raw_data, pred_data):
        return 1 - np.mean(np.abs((raw_data - pred_data) / raw_data), axis = 0)
    
    mape = mape(raw_d, pred_d) # mape 구함
    mape_ = list()
    mape_.append(pred_data[-1, 0]) #제일마지막 행(-1)의 0번째 열(날짜) -> 즉 해당 '날짜'를 어펜드 하는것! 값이 아님.
	#========================================================================================

	# try 블록 수행 중 오류가 발생하면 except 블록이 수행된다.================================= 
	#하지만 try 블록에서 오류가 발생하지 않는다면 except 블록은 수행되지 않는다.
    try:
        for m in mape:
            mape_.append(float(m)) #계산된 mape를 mape_에 어팬드 한다.
    except:
        for i in range(1, len(mape_field)): #mape를 계산할 실제값이 처음 5개에는 존재x -> 이 mape는 nan으로 처리
            mape_.append(np.nan)
	#==========================================================================================
    mape_ = [mape_]
    
    return mape_









# html페이지에 사용할 데이터 만들어주는 함수
#보간값에 맞는 필드딕셔너리 생성해서 디비에서 로드 하는 식.
def make_html_data(config_filepath = 'config.ini'):
    
    param = get_config(config_filepath)
    
    site = param['site']
    
    host = param['host']
    user = param['user']
    password = param['password']
    database = param['database']
    port = param['port']
    conn_inf = [host, user, password, database, port]
    
    raw_data_table = param['raw_data_table']
    data_table = param['data_table']
    pred_table = param['pred_table']
    
    raw_field_dict = param['raw_field']
    pred_field = param['pred_field']
    
    variables = param['vars']
    
    #각각의 보간 형태에 따른 필드 딕셔너리 생성 =================================
    intp_field_dict = make_field_info_suffix(raw_field_dict, '_intp') 
    nw_field_dict = make_field_info_suffix(raw_field_dict, '_nw')
    nw_intp_field_dict = make_field_info_suffix(raw_field_dict, '_nw_intp')
    pred_field_dict = make_pred_field_info(raw_field_dict, pred_field, pred = True)
    #==========================================================================
    
    #raw_data DB에서 로드    
    raw_data = load_data(conn_inf, site, raw_data_table, raw_field_dict)
    raw_data.columns = raw_field_dict.values()

    #각 보간값 DB에서 로드
    try:
        linear_intp_data = load_data(conn_inf, site, data_table, intp_field_dict)
        nw_data = load_data(conn_inf, site, data_table, nw_field_dict)
        nw_intp_data = load_data(conn_inf, site, data_table, nw_intp_field_dict)
        linear_intp_data.columns = intp_field_dict.values()
        nw_data.columns = nw_field_dict.values()
        nw_intp_data.columns = nw_intp_field_dict.values()        
        
    except:
        linear_intp_data = pd.DataFrame(np.zeros_like(raw_data))
        nw_data = pd.DataFrame(np.zeros_like(raw_data))
        nw_intp_data = pd.DataFrame(np.zeros_like(raw_data))
        linear_intp_data.columns = intp_field_dict.values()
        nw_data.columns = nw_field_dict.values()
        nw_intp_data.columns = nw_intp_field_dict.values()     
        linear_intp_data['Time'] = raw_data['Time'] #0으로 채워져 있던 값을 다시 날짜로 채움.
        nw_data['Time'] = raw_data['Time']      
        nw_intp_data['Time'] = raw_data['Time']       

    # 예측값 DB에서 로드    
    try:
        pred_data = load_data(conn_inf, site, pred_table, pred_field_dict)
        pred_data.columns = pred_field_dict.values()
    except:
        pred_data = None

    #mape 값 계산
    mape_data = make_mape(conn_inf, raw_data_table, pred_table, raw_field_dict, pred_field_dict)
    mape_data = pd.DataFrame(mape_data)
    mape_data.columns = pred_field_dict.values()
    
    v_dict = dict()
    for i, v in enumerate([v for l in variables.values() for v in l]):
        v_dict[v] = i
    
    v_list = list(v_dict.keys())
    
    return raw_data, linear_intp_data, nw_data, nw_intp_data, pred_data, mape_data, variables, v_dict, v_list


