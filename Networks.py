from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.keras import Model
from tensorflow.python.keras import initializers
from tensorflow.python.keras import activations
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import random_ops
#from tensorflow.python.ops import state_ops
#from tensorflow.python.ops import variables
from tensorflow.python.framework import dtypes
#from tensorflow.python.framework import constant_op
#from tensorflow.python.ops import clip_ops
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import tensor_shape

from tensorflow.python.eager.backprop import GradientTape
from tensorflow.python.eager.def_function import function

from tensorflow.keras.optimizers.schedules import LearningRateSchedule

import db_utils as utils

import numpy as np
import time


def xavier_init(fan_in, fan_out, *, const = 1.0, dtype = dtypes.float32):
    
    k = const * math_ops.sqrt(6.0 / (fan_in + fan_out))
    
    return random_ops.random_uniform((fan_in, fan_out), minval = -k, maxval = k, dtype = dtype)


def sample_bernoulli(probs):
    
    activation = activations.get('relu')
    
    return activation(math_ops.sign(probs - random_ops.random_uniform(array_ops.shape(probs)))) 
#0보다 작으면0, 1이상은 1 그럼 쉐잎은 변화x


def sample_gaussian(x, sigma):
    
    return x + random_ops.random_normal(array_ops.shape(x), mean = 0.0, stddev = sigma, dtype = dtypes.float32)
    

class Schedule(LearningRateSchedule):
    
    def __init__(self, embedding_dim, start_step, warmup_steps = 2500):
      
        super(Schedule, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.embedding_dim = math_ops.cast(self.embedding_dim, dtypes.float32)
        
        self.start_step = start_step
        self.warmup_steps = warmup_steps
    
    
    def __call__(self, step):

        step += self.start_step
        
        value1 = math_ops.rsqrt(step)
        value2 = step * (self.warmup_steps ** -1.5)
        
        return math_ops.rsqrt(self.embedding_dim) * math_ops.minimum(value1, value2)



class dense(Layer): 
    
    def __init__(self, units, activation = None, initializer = 'glorot_uniform', trainable = True, dynamic = True):

        super(dense, self).__init__()

        self.units = units
        self.activation = activations.get(activation)
        self.initializer = initializers.get(initializer)
        

    def build(self, input_shape):
        
        last_dim = tensor_shape.dimension_value(input_shape[-1]) #input shape의 마지막 차원은 2임.(타임과 orl 2차원.)
        #fnn 에서 units = [15,15,3]이 for 돌면서 하나씩 하나 씩 옴 즉 kernerl의 쉐잎은 (2,15) (2,15) (2,3) 3층을 거쳐야 해서 3개가 필요한것으로 추정...
        self.kernel = self.add_weight('kernel',
                                      shape = (last_dim, self.units), 
                                      initializer = self.initializer,
                                      dtype = 'float32',
                                      trainable = True)
        
        self.bias = self.add_weight('bias',
                                    shape = (self.units), #마찬가지로 bias의 쉐입도 3층을 거쳐야해서 (15,) (15,) (3,)로 나올듯
                                    initializer = initializers.get('zeros'),
                                    dtype = 'float32',
                                    trainable = True)        
    
    def call(self, inputs): 
        
        outputs = math_ops.matmul(inputs, self.kernel) + self.bias 
        if self.activation is not None:
            
            outputs = self.activation(outputs)

        return outputs 


class fnn(Layer):
    def __init__(self, units, activations, initializers, optimizer, trainable = True, dynamic = True):
        super(fnn, self).__init__()

        if not isinstance(units, (list, tuple)): #units가 list인지 tuple인지 알아봄 (트루펄스로 나오는듯)
            
            units = [units]
            
        if not isinstance(activations, (list, tuple)):
            
            activations = [activations]
            
        if not isinstance(initializers, (list, tuple)):
            
            initializers = [initializers]

        if activations[-1] != 'linear':

            raise ValueError("Error: output activation type must be None or 'linear'")

        self.optimizer = optimizer            
            
        self.layers = list()
        
        for unit, activation, initializer in zip(units, activations, initializers):
            
            self.layers.append(dense(unit, activation, initializer)) 
            

    def call(self, inputs):

        for layer in self.layers:
            inputs = layer(inputs) #layers는 dense의 객체라서 inputs 받는것.
            #이 레이어스가 inputs 받아서 차원 맞춰놓은것임. 레이어스가 3개 나올꺼니까 for 돌려서 1층 2층 3층 순차적으로 객체를 받음.
            #layers는 dense 클래스의 객체임. 따라서 그 객체에다가 inputs 를 넣은것임.
           
        
        return inputs


    def loss_function(self, targets, predictions):
        #평균제곱오차 계산인거같음

        loss = math_ops.square(targets - predictions)
    
        return math_ops.reduce_mean(loss) #결국 하나의 값으로 계산됨

    
    @function
    def train_step(self, inputs, targets): #train 함수에 의해 inputs : batch_X, targets : batch_Y1
        
        with GradientTape() as tape:
            
            predictions = self(inputs) #fnn 클래스 이용해서 inputs를 통해 key예측값 뱉은거임. (call에서 나온 값을 predictions로 받겠다는 뜻)
            #like a=fnn(인수) a(inputs) 이렇게 쓰는거 즉 fnn(inputs) 
            loss = self.loss_function(targets, predictions) #key랑 key예측값이랑 loss_function 함수로 loss계산
            
        gradients = tape.gradient(loss, self.trainable_variables) 
        #주어진 입력 변수에 대한 연산의 그래디언트(gradient)를 계산하는 것  
        #trainable_variables : 변수목록 반환
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables)) 
        #(그래디언트, 변수) 변수에 그래디언트 적용. (오차역전파(Backpropagation) - weight 업데이트)
        
        return predictions, loss

    def train(self, train_data, epochs, TAKE, EPS = 1E-3): 
        #손실함수가 어떻게 변화하는지 살펴보는게 주 기능임.
        #손실함수의 계산이나 미분방향으로 업뎃은 train_step에서.
        
        print('learing feed forward neural network...')
        print('-' * 50)
        
        previous_epoch_loss = 1E+32
        
        for epoch in range(epochs):
            epoch_loss = 0.
            for e, (batch_X, batch_Y1, batch_Y2) in enumerate(train_data.take(TAKE)): #e는 반복횟수 카운팅됨.
                # batch_X (배치, 인풋차원), batch_Y1(배치, 키팩터 차원), batch_Y2(배치, 아웃풋 차원)
                predictiions, loss = self.train_step(batch_X, batch_Y1) #predic, loss 둘다 받아도 loss밖에 안씀.
                epoch_loss += loss
                
            epoch_loss /= TAKE #한 텀에서 loss의 평균. why? 작은 for가 TAKE 만큼 돌기 때문에.
            
            print('%d: %.4f' % ((epoch + 1), epoch_loss))
            
            if abs(previous_epoch_loss - epoch_loss) < EPS * previous_epoch_loss:
                
                break
            
            previous_epoch_loss = epoch_loss    
            
        print('-' * 50)
        print('Done.')
        print('-' * 50)
            
        
        
    
class gbrbm(Layer):
    
    def __init__(self, unit, activation, initializer, optimizer, trainable = True, dynamic = True):
        
        super(gbrbm, self).__init__()
        
        self.unit = unit
        self.activation = activations.get(activation)
        self.initializer = initializers.get(initializer)
        self.optimizer = optimizer
        
        
    def build(self, input_shape):
        
        last_dim = tensor_shape.dimension_value(input_shape[-1])

        self.W = self.add_weight('W',
                                 shape = (last_dim, self.unit), #rbm에서 layers쌓을때 첫번재 for에서 사용하는게 gbrbm. 그래서 unit이 [15,15]로 나오는게 아니고 15임. (첫번째 for)
                                 initializer = self.initializer, #그래서 (3(키팩터 차원=데이터 차원), rbm 유닛1) 쉐잎.
                                 dtype = 'float32',
                                 trainable = True)
        
        self.hbias = self.add_weight('hbias',
                                     shape = (self.unit), #(유닛1,)
                                     initializer = initializers.get('ones'),
                                     dtype = 'float32',
                                     trainable = True)        
        
        self.vbias = self.add_weight('vbias',
                                     shape = (last_dim), #(3,)
                                     initializer = initializers.get('ones'),
                                     dtype = 'float32',
                                     trainable = True)        

    def call(self, inputs):

        self.hiddens = self.activation(math_ops.matmul(inputs, self.W) + self.hbias) #(배치, rbm 유닛1)
        
        self.visibles = math_ops.matmul(self.hiddens, array_ops.transpose(self.W)) + self.vbias #(배치,3)
        
        return self.hiddens

         
    def forward(self, visible): #여기서 visible은 loss_function에서 받은 inputs 즉 key
        
        hiddens = self.activation(math_ops.matmul(visible, self.W) + self.hbias) #(배치, rbm 유닛1)
        
        return hiddens
            

    def backward(self, hiddens):
        
        visibles = math_ops.matmul(sample_bernoulli(hiddens), array_ops.transpose(self.W)) + self.vbias
        
        return visibles # (배치,(3=데이터 차원))
    

    def loss_function(self, inputs):

        hiddens = self.forward(inputs) #(배치, rbm 유닛1)
        visibles = self.backward(hiddens) # (배치, 데이터 차원)
        
        return math_ops.reduce_mean(math_ops.square(inputs - visibles))


    @function
    def train_step(self, inputs): #train에서 받아온 inputs는 keyfactor 여서 (배치,3(데이터 차원)) 쉐잎임. 
        
        with GradientTape() as tape:
            
            _ = self(inputs) 
            # 아 call에서 받은 hiddens이 필요없대 어차피 inputs - visivles만 필요.
            #gbrbm을 이용해 포워드 백워드 로스 계산할거라서 객체 만듦
            loss = self.loss_function(inputs) 
            
        gradients = tape.gradient(loss, self.trainable_variables)    
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        return loss  
    
    
    def train(self, train_data, epochs, TAKE, EPS = 1E-3):

        print('learning gbrbm layer....')
        print('-' * 50)

        previous_epoch_loss = 1E+32
        
        for epoch in range(epochs):
            epoch_loss = 0.
            for e, (_, inputs, _) in enumerate(train_data.take(TAKE)): #세 덩어리중 가운데만 받는다는건 key만 받는다는거지..?
            #input : (배치, 3 (keyfactor 변수 갯수임))
                loss = self.train_step(inputs)
                epoch_loss += loss
                
            epoch_loss /= TAKE
            
            print('%d: %.4f' % ((epoch + 1), epoch_loss))
           
            if abs(previous_epoch_loss - epoch_loss) < EPS * previous_epoch_loss:
                
                break
            
            previous_epoch_loss = epoch_loss    
    
        print('-' * 50)
        print('Done.')
        print('-' * 50)


class bbrbm(Layer):
    
    def __init__(self, unit, activation, initializer, optimizer, trainable = True, dynamic = True):
        
        super(bbrbm, self).__init__()
        
        self.unit = unit
        self.activation = activations.get(activation)
        self.initializer = initializers.get(initializer)
        self.optimizer = optimizer
        
        
    def build(self, input_shape):
        
        last_dim = tensor_shape.dimension_value(input_shape[-1])

        self.W = self.add_weight('W',
                                 shape = (last_dim, self.unit), # 15(유닛1),15(유닛2)
                                 initializer = self.initializer,
                                 dtype = 'float32',
                                 trainable = True)
        
        self.hbias = self.add_weight('hbias',
                                     shape = (self.unit), # (유닛2,)
                                     initializer = initializers.get('ones'),
                                     dtype = 'float32',
                                     trainable = True)        
        
        self.vbias = self.add_weight('vbias',
                                     shape = (last_dim), # (데이터차원,)
                                     initializer = initializers.get('ones'),
                                     dtype = 'float32',
                                     trainable = True)        

    def call(self, inputs):

        self.hiddens = self.activation(math_ops.matmul(inputs, self.W) + self.hbias) #배치,유닛2
        
        self.visibles = self.activation(math_ops.matmul(self.hiddens, array_ops.transpose(self.W)) + self.vbias) #
        
        return self.hiddens


    def forward(self, visible):
        
        hiddens = self.activation(math_ops.matmul(visible, self.W) + self.hbias) # (배치, 유닛2)
        
        return hiddens
            

    def backward(self, hiddens):
        
        visibles_ = math_ops.matmul(sample_bernoulli(hiddens), array_ops.transpose(self.W)) + self.vbias
        visibles = self.activation(visibles_) # (배치,유닛1)
        
        return visibles # (배치,유닛1)
    

    def loss_function(self, inputs):

        hiddens = self.forward(inputs) # # (배치, 유닛2)
        visibles = self.backward(hiddens) ## (배치,유닛1)
        
        return math_ops.reduce_mean(math_ops.square(inputs - visibles)) #결국엔 하나의 값


    @function
    def train_step(self, inputs):
        
        with GradientTape() as tape:
            
            _ = self(inputs) #bbrbm을 이용해 포워드 백워드 로스 계산할거라서 객체 만듦
            loss = self.loss_function(inputs) #여기 inputs는 (배치,유닛1(15))가 맞음.
            
        gradients = tape.gradient(loss, self.trainable_variables)    
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        return loss  
    
    
    def train(self, train_data, epochs, TAKE, EPS = 1E-3): #여기서 받는 train_data는 (None, 유닛1) None = 배치로 생각하면될듯
    
        print('learning bbrbm layer....')
        print('-' * 50)

        previous_epoch_loss = 1E+32
        
        for epoch in range(epochs):
            epoch_loss = 0.
            for e, inputs in enumerate(train_data.take(TAKE)):
                               
                loss = self.train_step(inputs) #여기서 inputs는 (배치, 유닛1) -> rbm의 train 두번째 for에서.
                epoch_loss += loss
                
            epoch_loss /= TAKE
            
            print('%d: %.4f' % ((epoch + 1), epoch_loss))
            
            if abs(previous_epoch_loss - epoch_loss) < EPS * previous_epoch_loss:
                
                break
            
            previous_epoch_loss = epoch_loss    
        
        print('-' * 50)
        print('Done.')
        print('-' * 50)
        
    
    
class rbm(Layer):

    def __init__(self, units, activations, initializers, optimizer, trainable = True, dynamic = True):
        super(rbm, self).__init__()
        
        
        if not isinstance(units, (list, tuple)):
            
            units = [units]
            
        if not isinstance(activations, (list, tuple)):
            
            activations = [activations]
            
        if not isinstance(initializers, (list, tuple)):
            
            initializers = [initializers]

        self.optimizer = optimizer
            
        self.layers = list()
        
        for layer_idx, (unit, activation, initializer) in enumerate(zip(units, activations, initializers)):

            if layer_idx < 1:            
                self.layers.append(gbrbm(unit, activation, initializer, optimizer)) # 15,15
            else:
                self.layers.append(bbrbm(unit, activation, initializer, optimizer))
        
            
    def call(self, inputs):

        for layer in self.layers:
            
            inputs = layer(inputs)
        
        return inputs
    
    
    def get_train_hiddens(self, train_data, layer, TAKE):
        
        outputs = list()
        
        for _, inputs, _ in train_data.take(TAKE):
            #inputs (배치, 3(키팩터 변수갯수.))
            
            hiddens = layer.forward(inputs) 
            #여기서 받은 inputs는 keyfactor가 맞음 -> gbrbm의 forword확인하면 될듯
            #why? 이 함수는 layer_idx < 1일 때만 시행되는데 그 경우는 gbrbm이 시행되는 경우임. -> (배치,rbm 유닛1) 쉐잎.
            
            for opt in hiddens:
                outputs.append(opt)
        
        return outputs # (유닛1,)가 배치개 나옴.
            
            

    def train(self, train_data, batch_size, epochs, TAKE, EPS = 1E-4):

        for layer_idx, layer in enumerate(self.layers):
        
            layer.train(train_data, epochs, TAKE) 
            #여기서 첫번째 layer는 gbrbm, 두번째 layer는 bbrbm.
            #첫번째 for에서 gbrbm의 train 함수 시행 -> loss계산이 되겠지
            #두번째 for에서 bbrbm의 train 함수 시행. 이때 들어오는 train_data는 아래 if문을 통해 갱신된 train_data. -> loss 계산.
            
            if layer_idx < 1: 

                outputs = self.get_train_hiddens(train_data, layer, TAKE) # (유닛1,)가 배치개 나옴.
            
                train_data = dataset_ops.DatasetV2.from_tensor_slices(outputs) #(유닛1,)
                train_data = train_data.batch(batch_size).repeat()#.shuffle(len(train_set)) #(배치,유닛1)

            
class Model_(Model):
    
    def __init__(self, num_out, pre_optimizer, fine_optimizer):

        super(Model_, self).__init__()
        
        params = utils.get_config()
        
        fnn_units = params['fnn_units']
        fnn_activations = params['fnn_activations']
        fnn_initializers = params['fnn_initializers']
        
        rbm_units = params['rbm_units']
        rbm_activations = params['rbm_activations']
        rbm_initializers = params['rbm_initializers']
        
        self.pre_optimizer = pre_optimizer
        self.fine_optimizer = fine_optimizer
        
        self.fnn_layer = fnn(fnn_units, fnn_activations, fnn_initializers, pre_optimizer)
        
        self.rbm_layer = rbm(rbm_units, rbm_activations, rbm_initializers, pre_optimizer)
        
        self.dns = dense(num_out, 'linear', 'zeros')
        
        #self.alpha = variables.Variable(initial_value = 0.5, trainable = True, dtype = dtypes.float32)

        
    def call(self, inputs, key_inputs): #여기가 예측. (학습이랑 받는 인수가 다른것 확인.)

        key_predictions = self.fnn_layer(inputs) #input으로 key 예측함. 
        #rbm_key_outputs = self.rbm_layer(key_predictions)
        
        rbm_outputs = self.rbm_layer(key_inputs)
        out_predictions = self.dns(rbm_outputs) # key로 output 예측함 (???)
        
        rbm_key_outputs = self.rbm_layer(key_predictions)#fnn을 이용해 input으로 key예측한 값을 rbm을 사용해서 output 예측함.
        input_predictions = self.dns(rbm_key_outputs)
        
        return key_predictions, out_predictions, input_predictions
        #근데 여기서 DB에 업로드할때 input_predictions는 업로드 안하더라
        
        
    def pre_train(self, train_data, batch_size, epochs, TAKE, EPS = 1E-4): #학습시키는거야.
        
        print('-' * 50)
        print('start pre-training...')
        print('-' * 50)
        self.fnn_layer.train(train_data, epochs, TAKE, EPS)        
        self.rbm_layer.train(train_data, batch_size, epochs, TAKE, EPS)
        
        
    def loss_function(self, inputs, key_inputs, targets):
        
        key_predictions, out_predictions, input_predictions = self(inputs, key_inputs) #Model_.call을 통해 받는 값.
        
        ##reduce_mean : 특정 차원을 제거하고 평균을 구한다.
        key_loss = math_ops.reduce_mean(math_ops.square(key_inputs - key_predictions)) 
        out_loss = math_ops.reduce_mean(math_ops.square(targets - out_predictions))
        input_out_loss = math_ops.reduce_mean(math_ops.square(targets - input_predictions))
        
        #loss = (1. - self.alpha) * key_loss  + self.alpha * out_loss + 0.5 * math_ops.square(self.alpha)
        #loss = (key_loss + out_loss + out_loss + input_out_loss) / 3.
        loss = 0.75 * key_loss + 0.25 * out_loss #+ 0. * input_out_loss
        
        return key_predictions, out_predictions, loss, key_loss, out_loss, input_out_loss
    
    
    @function
    def train_step(self, inputs, key_inputs, targets):
        
        with GradientTape() as tape:
            
            key_predictions, out_predictions, loss, k, o, io = self.loss_function(inputs, key_inputs, targets)
            # 저 값들 자체가 loss_function을 통해서 계산된 loss.
            
        weights = self.trainable_variables # trainable_variables : 변수목록 반환
        train_weights = list()
        for w in weights:
            if 'vbias' not in w.name:
                train_weights.append(w)
                
        gradients = tape.gradient(loss, train_weights) #주어진 입력변수에 대한 연산의 그래디언트 계산
        self.fine_optimizer.apply_gradients(zip(gradients, train_weights)) 
        #(그래디언트,변수) : 변수에 그래디언트 적용 (가중치 업데이트)
        
        return key_predictions, out_predictions, loss, k, o, io
    

    def mape(self, raw_data, pred_data):
        
        return 1 - np.mean(np.abs(raw_data - pred_data) / np.abs(raw_data + 1E-8), axis = 0)
 
    
    def fine_tuning(self, train_data, epochs, TAKE, restore, EPS = 1E-4):
        
        print('-' * 50)
        print('network fine-tuning...')
        print('-' * 50)
        
        previous_epoch_loss = 1E+32
        #previous_min_mape = 1E+32
        
        best_loss = 1E+32
       # best_mape = 1E-32
        
        step = 0
        patience = 100
        
        for epoch in range(epochs):
            epoch_loss = 0.
            epoch_key_mape = 0.
            epoch_out_mape = 0.
            for e, (batch_X, batch_Y1, batch_Y2) in enumerate(train_data.take(TAKE)):
                #batch_X : (배치, 인풋차원), batch_Y1 : (배치, 키팩터 차원), batch_Y2 : (배치, 아웃풋 차원)
                #batch_X+5일이 batch_Y1 , batch_Y2날짜와 동일.
                key_predictions, out_predictions, loss, k, o, io = self.train_step(batch_X, batch_Y1, batch_Y2)
                
                #print((k.numpy(), o.numpy(), io.numpy()))
                #print(batch_Y1)
                #print(key_predictions)
                #print(self.mape(batch_Y1, key_predictions))
                #print(self.mape(batch_Y2, out_predictions))
                
                epoch_loss += loss
                epoch_key_mape += self.mape(batch_Y1, key_predictions)
                epoch_out_mape += self.mape(batch_Y2, out_predictions)
                
            epoch_loss /= TAKE  
            epoch_key_mape /= TAKE
            epoch_out_mape /= TAKE
            
            #print(epoch_key_mape)
            #print(epoch_out_mape)
            
            if best_loss < epoch_loss:
                step += 1
                if step > patience:
                    self.load_weights(restore)
                    break
            else:
                step = 0
                best_loss = epoch_loss
                
                time.sleep(0.05)
                
                self.save_weights(restore)
                
            
            
            mean_mape = np.mean((np.mean(epoch_key_mape), np.mean(epoch_out_mape)))
            print('%d: %.5f (%.5f)' % ((epoch + 1), epoch_loss, mean_mape))
            #print(mean_mape)
            #if ((np.min(epoch_out_mape) > 0.90 or abs(previous_min_mape - np.min(epoch_out_mape) < EPS * previous_min_mape)
            #    and np.min(epoch_key_mape) > 0.75)
            #    and abs(previous_epoch_loss - epoch_loss) < EPS * previous_epoch_loss):
            #if abs(previous_epoch_loss - epoch_loss) < EPS * previous_epoch_loss: # and mean_mape > 0.50:

            #    break
            
            previous_epoch_loss = epoch_loss
            #previous_min_mape = np.min(epoch_out_mape)
            
        print('-' * 50)
        print('Done.')
        print('-' * 50)
        
