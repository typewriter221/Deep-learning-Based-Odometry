import tfquaternion as tfq
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Bidirectional, LSTM, Dropout, Dense, Input, Layer, Conv1D, MaxPooling1D, concatenate
from keras.initializers import Constant
from keras.optimizers import Adam
from keras.losses import mean_absolute_error
from keras import backend as K
from tensorflow.compat.v1.keras.layers import CuDNNLSTM



import numpy as np
import pandas as pd
import os, datetime
import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *


def quaternion_phi_3_error(y_true, y_pred):
    return tf.acos(K.abs(K.batch_dot(y_true, K.l2_normalize(y_pred, axis=-1), axes=-1)))


def quaternion_phi_4_error(y_true, y_pred):
    return 1 - K.abs(K.batch_dot(y_true, K.l2_normalize(y_pred, axis=-1), axes=-1))


def quaternion_log_phi_4_error(y_true, y_pred):
    return K.log(1e-4 + quaternion_phi_4_error(y_true, y_pred))


def quat_mult_error(y_true, y_pred):
    q_hat = tfq.Quaternion(y_true)
    q = tfq.Quaternion(y_pred).normalized()
    q_prod = q * q_hat.conjugate()
    w, x, y, z = tf.split(q_prod, num_or_size_splits=4, axis=-1)
    return tf.abs(tf.multiply(2.0, tf.concat(values=[x, y, z], axis=-1)))


def quaternion_mean_multiplicative_error(y_true, y_pred):
    return tf.reduce_mean(quat_mult_error(y_true, y_pred))


# Custom loss layer
class CustomMultiLossLayer(Layer):
    def __init__(self, nb_outputs=2, **kwargs):
    #def __init__(self, nb_outputs=3, **kwargs):
        self.nb_outputs = nb_outputs
        self.is_placeholder = True
        super(CustomMultiLossLayer, self).__init__(**kwargs)
    
    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'nb_outputs': self.nb_outputs,
            # 'is_placeholder': self.is_placeholder,
            # 'log_vars':self.log_vars
            
        })
        return config
        
    def build(self, input_shape=None):
        # initialise log_vars
        self.log_vars = []
        for i in range(self.nb_outputs):
            self.log_vars += [self.add_weight(name='log_var' + str(i), shape=(1,),
                                              initializer=Constant(0.), trainable=True)]
        super(CustomMultiLossLayer, self).build(input_shape)

    def multi_loss(self, ys_true, ys_pred):
        assert len(ys_true) == self.nb_outputs and len(ys_pred) == self.nb_outputs
        loss = 0

        #for y_true, y_pred, log_var in zip(ys_true, ys_pred, self.log_vars):
        #    precision = K.exp(-log_var[0])
        #    loss += K.sum(precision * (y_true - y_pred)**2., -1) + log_var[0]

        precision = K.exp(-self.log_vars[0][0])
        loss += precision * mean_absolute_error(ys_true[0], ys_pred[0]) + self.log_vars[0][0]
        precision = K.exp(-self.log_vars[1][0])
        loss += precision * quaternion_mean_multiplicative_error(ys_true[1], ys_pred[1]) + self.log_vars[1][0]
        #loss += precision * quaternion_phi_4_error(ys_true[1], ys_pred[1]) + self.log_vars[1][0]

        return K.mean(loss)

    def call(self, inputs):
        ys_true = inputs[:self.nb_outputs]
        ys_pred = inputs[self.nb_outputs:]
        loss = self.multi_loss(ys_true, ys_pred)
        self.add_loss(loss, inputs=inputs)
        # We won't actually use the output.
        return K.concatenate(inputs, -1)

########################ATTENTION MODEL############################################
class Time2Vector(Layer):
  def __init__(self, seq_len = 200, **kwargs):
    super(Time2Vector, self).__init__()
    self.seq_len = seq_len

  def build(self, input_shape):
    '''Initialize weights and biases with shape (batch, seq_len)'''
    self.weights_linear = self.add_weight(name='weight_linear',
                                shape=(int(self.seq_len),),
                                initializer='uniform',
                                trainable=True)
    
    self.bias_linear = self.add_weight(name='bias_linear',
                                shape=(int(self.seq_len),),
                                initializer='uniform',
                                trainable=True)
    
    self.weights_periodic = self.add_weight(name='weight_periodic',
                                shape=(int(self.seq_len),),
                                initializer='uniform',
                                trainable=True)

    self.bias_periodic = self.add_weight(name='bias_periodic',
                                shape=(int(self.seq_len),),
                                initializer='uniform',
                                trainable=True)
    '''For Second Data'''
    self.weights_linear1 = self.add_weight(name='weight_linear1',
                                shape=(int(self.seq_len),),
                                initializer='uniform',
                                trainable=True)
    
    self.bias_linear1 = self.add_weight(name='bias_linear1',
                                shape=(int(self.seq_len),),
                                initializer='uniform',
                                trainable=True)
    
    self.weights_periodic1 = self.add_weight(name='weight_periodic1',
                                shape=(int(self.seq_len),),
                                initializer='uniform',
                                trainable=True)

    self.bias_periodic1 = self.add_weight(name='bias_periodic1',
                                shape=(int(self.seq_len),),
                                initializer='uniform',
                                trainable=True)


  def call(self, x):
    '''Calculate linear and periodic time features'''
    # x = tf.math.reduce_mean(x[:,:,:4], axis=-1) 
    # time_linear = self.weights_linear * x + self.bias_linear # Linear time feature
    # time_linear = tf.expand_dims(time_linear, axis=-1) # Add dimension (batch, seq_len, 1)
    
    # time_periodic = tf.math.sin(tf.multiply(x, self.weights_periodic) + self.bias_periodic)
    # time_periodic = tf.expand_dims(time_periodic, axis=-1) # Add dimension (batch, seq_len, 1)
    # return tf.concat([time_linear, time_periodic], axis=-1) # shape = (batch, seq_len, 2)
   
    ''' x1 = dtat1 x2= data2'''
    x1 = tf.math.reduce_mean(x[:,:,:3], axis=-1) 
    x2 = tf.math.reduce_mean(x[:,:,3:], axis=-1) 

    time_linear = self.weights_linear * x1 + self.bias_linear # Linear time feature
    time_linear = tf.expand_dims(time_linear, axis=-1) # Add dimension (batch, seq_len, 1)
    
    time_linear1 = self.weights_linear1 * x2 + self.bias_linear1 # Linear time feature
    time_linear1 = tf.expand_dims(time_linear1, axis=-1) # Add dimension (batch, seq_len, 1)
    
    time_periodic = tf.math.sin(tf.multiply(x1, self.weights_periodic) + self.bias_periodic)
    time_periodic = tf.expand_dims(time_periodic, axis=-1) # Add dimension (batch, seq_len, 1)
    
    time_periodic1 = tf.math.sin(tf.multiply(x2, self.weights_periodic1) + self.bias_periodic1)
    time_periodic1 = tf.expand_dims(time_periodic1, axis=-1) # Add dimension (batch, seq_len, 1)
    
    
    return tf.concat([time_linear, time_periodic, time_linear1, time_periodic1], axis=-1) # shape = (batch, seq_len, 4)
    
  def get_config(self): # Needed for saving and loading model with custom layer
    config = super().get_config().copy()
    config.update({'seq_len': self.seq_len})
    return config


class SingleAttention(Layer):
  def __init__(self, d_k = 256, d_v = 256):
    super(SingleAttention, self).__init__()
    self.d_k = d_k
    self.d_v = d_v

  def build(self, input_shape):
    self.query = Dense(self.d_k, 
                       input_shape=input_shape, 
                       kernel_initializer='glorot_uniform', 
                       bias_initializer='glorot_uniform')
    
    self.key = Dense(self.d_k, 
                     input_shape=input_shape, 
                     kernel_initializer='glorot_uniform', 
                     bias_initializer='glorot_uniform')
    
    self.value = Dense(self.d_v, 
                       input_shape=input_shape, 
                       kernel_initializer='glorot_uniform', 
                       bias_initializer='glorot_uniform')

  def call(self, inputs): # inputs = (in_seq, in_seq, in_seq)
    q = self.query(inputs[0])
    k = self.key(inputs[1])

    attn_weights = tf.matmul(q, k, transpose_b=True)
    attn_weights = tf.map_fn(lambda x: x/np.sqrt(self.d_k), attn_weights)
    attn_weights = tf.nn.softmax(attn_weights, axis=-1)
    
    v = self.value(inputs[2])
    attn_out = tf.matmul(attn_weights, v)
    return attn_out

class MultiAttention(Layer):
  def __init__(self, d_k = 256, d_v = 256, n_heads = 12):
    super(MultiAttention, self).__init__()
    self.d_k = d_k
    self.d_v = d_v
    self.n_heads = n_heads
    self.attn_heads = list()

  def build(self, input_shape):
    for n in range(self.n_heads):
      self.attn_heads.append(SingleAttention(self.d_k, self.d_v))  
    self.linear = Dense(input_shape[0][-1], input_shape=input_shape, kernel_initializer='glorot_uniform', bias_initializer='glorot_uniform')

  def call(self, inputs):
    attn = [self.attn_heads[i](inputs) for i in range(self.n_heads)]
    concat_attn = tf.concat(attn, axis=-1)
    multi_linear = self.linear(concat_attn)
    return multi_linear 

class TransformerEncoder(Layer):
  def __init__(self, d_k = 256, d_v = 256, n_heads = 12, ff_dim = 256, dropout=0.1, **kwargs):
    super(TransformerEncoder, self).__init__()
    self.d_k = d_k
    self.d_v = d_v
    self.n_heads = n_heads
    self.ff_dim = ff_dim
    self.attn_heads = list()
    self.dropout_rate = dropout
  def get_config(self):

        config = super().get_config().copy()
        config.update({
            'd_k': self.d_k,
            'd_v': self.d_v,
            'n_heads': self.n_heads,
            'ff_dim': self.ff_dim,
            'dropout': self.dropout_rate
        })
        return config

  def build(self, input_shape):
    self.attn_multi = MultiAttention(self.d_k, self.d_v, self.n_heads)
    self.attn_dropout = Dropout(self.dropout_rate)
    self.attn_normalize = LayerNormalization(input_shape=input_shape, epsilon=1e-6)

    self.ff_conv1D_1 = Conv1D(filters=self.ff_dim, kernel_size=1, activation='relu')
    self.ff_conv1D_2 = Conv1D(filters=input_shape[0][-1], kernel_size=1) # input_shape[0]=(batch, seq_len, 8), input_shape[0][-1]=8 
    self.ff_dropout = Dropout(self.dropout_rate)
    self.ff_normalize = LayerNormalization(input_shape=input_shape, epsilon=1e-6)    
  
  def call(self, inputs): # inputs = (in_seq, in_seq, in_seq)
    attn_layer = self.attn_multi(inputs)
    attn_layer = self.attn_dropout(attn_layer)
    attn_layer = self.attn_normalize(inputs[0] + attn_layer)

    ff_layer = self.ff_conv1D_1(attn_layer)
    ff_layer = self.ff_conv1D_2(ff_layer)
    ff_layer = self.ff_dropout(ff_layer)
    ff_layer = self.ff_normalize(inputs[0] + ff_layer)
    return ff_layer 


def create_pred_model_6d_quat(window_size=200):
    #inp = Input((window_size, 6), name='inp')
    #lstm1 = Bidirectional(CuDNNLSTM(128, return_sequences=True))(inp)
    x1 = Input((window_size, 3), name='x1')
    x2 = Input((window_size, 3), name='x2')
    # convA1 = Conv1D(128, 11)(x1)
    # convA2 = Conv1D(128, 11)(convA1)
    # poolA = MaxPooling1D(3)(convA2)
    # convB1 = Conv1D(128, 11)(x2)
    # convB2 = Conv1D(128, 11)(convB1)
    # poolB = MaxPooling1D(3)(convB2)
    in_seq= concatenate([x1, x2])
    batch_size = 32
    seq_len = 200

    d_k = 256
    d_v = 256
    n_heads = 12
    ff_dim = 256
    '''Initialize time and transformer layers'''
    time_embedding = Time2Vector(seq_len)
    attn_layer1 = TransformerEncoder(d_k, d_v, n_heads, ff_dim)
    attn_layer2 = TransformerEncoder(d_k, d_v, n_heads, ff_dim)
    attn_layer3 = TransformerEncoder(d_k, d_v, n_heads, ff_dim)

    '''Construct model'''
    # in_seq = Input(shape=(seq_len, 5))
    x = time_embedding(in_seq)
    # x = in_seq
    x = concatenate([in_seq, x],axis=-1)
    x = attn_layer1((x, x, x))
    x = attn_layer2((x, x, x))
    x = attn_layer3((x, x, x))
    x = GlobalAveragePooling1D(data_format='channels_first')(x)
    x = Dropout(0.1)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.1)(x)

    # lstm1 = Bidirectional(CuDNNLSTM(128, return_sequences=True))(AB)
    # drop1 = Dropout(0.25)(lstm1)
    # lstm2 = Bidirectional(CuDNNLSTM(128))(drop1)
    # drop2 = Dropout(0.25)(lstm2)    
    # y1_pred = Dense(3)(drop2)
    # y2_pred = Dense(4)(drop2)
    y1_pred = Dense(3)(x)
    y2_pred = Dense(4)(x)
    

    #model = Model(inp, [y1_pred, y2_pred])
    model = Model([x1, x2], [y1_pred, y2_pred])

    model.summary()
    
    return model

#########################################################################################################
def create_train_model_6d_quat(pred_model, window_size=200):
    #inp = Input(shape=(window_size, 6), name='inp')
    #y1_pred, y2_pred = pred_model(inp)
    x1 = Input((window_size, 3), name='x1')
    x2 = Input((window_size, 3), name='x2')
    y1_pred, y2_pred = pred_model([x1, x2])
    y1_true = Input(shape=(3,), name='y1_true')
    y2_true = Input(shape=(4,), name='y2_true')
    out = CustomMultiLossLayer(nb_outputs=2)([y1_true, y2_true, y1_pred, y2_pred])
    #train_model = Model([inp, y1_true, y2_true], out)
    train_model = Model([x1, x2, y1_true, y2_true], out)
    train_model.summary()
    return train_model


def create_pred_model_3d(window_size=200):
    #inp = Input((window_size, 6), name='inp')
    #lstm1 = Bidirectional(CuDNNLSTM(128, return_sequences=True))(inp)
    x1 = Input((window_size, 3), name='x1')
    x2 = Input((window_size, 3), name='x2')
    convA1 = Conv1D(128, 11)(x1)
    convA2 = Conv1D(128, 11)(convA1)
    poolA = MaxPooling1D(3)(convA2)
    convB1 = Conv1D(128, 11)(x2)
    convB2 = Conv1D(128, 11)(convB1)
    poolB = MaxPooling1D(3)(convB2)
    AB = concatenate([poolA, poolB])
    lstm1 = Bidirectional(CuDNNLSTM(128, return_sequences=True))(AB)
    drop1 = Dropout(0.25)(lstm1)
    lstm2 = Bidirectional(CuDNNLSTM(128))(drop1)
    drop2 = Dropout(0.25)(lstm2)
    y1_pred = Dense(1)(drop2)
    y2_pred = Dense(1)(drop2)
    y3_pred = Dense(1)(drop2)

    #model = Model(inp, [y1_pred, y2_pred, y3_pred])
    model = Model([x1, x2], [y1_pred, y2_pred, y3_pred])

    model.summary()
    
    return model


def create_train_model_3d(pred_model, window_size=200):
    #inp = Input(shape=(window_size, 6), name='inp')
    #y1_pred, y2_pred, y3_pred = pred_model(inp)
    x1 = Input((window_size, 3), name='x1')
    x2 = Input((window_size, 3), name='x2')
    y1_pred, y2_pred, y3_pred = pred_model([x1, x2])
    y1_true = Input(shape=(1,), name='y1_true')
    y2_true = Input(shape=(1,), name='y2_true')
    y3_true = Input(shape=(1,), name='y3_true')
    out = CustomMultiLossLayer(nb_outputs=3)([y1_true, y2_true, y3_true, y1_pred, y2_pred, y3_pred])
    #train_model = Model([inp, y1_true, y2_true, y3_true], out)
    train_model = Model([x1, x2, y1_true, y2_true, y3_true], out)
    train_model.summary()
    return train_model


def create_model_6d_rvec(window_size=200):
    input_gyro_acc = Input((window_size, 6))
    lstm1 = Bidirectional(CuDNNLSTM(128, return_sequences=True))(input_gyro_acc)    
    drop1 = Dropout(0.25)(lstm1)
    lstm2 = Bidirectional(CuDNNLSTM(128))(drop1)    
    drop2 = Dropout(0.25)(lstm2)    
    output_delta_rvec = Dense(3)(drop2)
    output_delta_tvec = Dense(3)(drop2)

    model = Model(inputs = input_gyro_acc, outputs = [output_delta_rvec, output_delta_tvec])
    model.summary()
    model.compile(optimizer = Adam(0.0001), loss = 'mean_squared_error')
    
    return model


def create_model_6d_quat(window_size=200):
    input_gyro_acc = Input((window_size, 6))
    lstm1 = Bidirectional(CuDNNLSTM(128, return_sequences=True))(input_gyro_acc)    
    drop1 = Dropout(0.25)(lstm1)
    lstm2 = Bidirectional(CuDNNLSTM(128))(drop1)    
    drop2 = Dropout(0.25)(lstm2)    
    output_delta_p = Dense(3)(drop2)
    output_delta_q = Dense(4)(drop2)

    model = Model(inputs = input_gyro_acc, outputs = [output_delta_p, output_delta_q])
    model.summary()
    #model.compile(optimizer = Adam(0.0001), loss = 'mean_squared_error')
    model.compile(optimizer = Adam(0.0001), loss = ['mean_absolute_error', quaternion_mean_multiplicative_error])
    #model.compile(optimizer = Adam(0.0001), loss = ['mean_absolute_error', quaternion_phi_4_error])
    
    return model


def create_model_3d(window_size=200):
    input_gyro_acc = Input((window_size, 6))
    lstm1 = Bidirectional(CuDNNLSTM(128, return_sequences=True))(input_gyro_acc)    
    drop1 = Dropout(0.25)(lstm1)
    lstm2 = Bidirectional(CuDNNLSTM(128))(drop1)    
    drop2 = Dropout(0.25)(lstm2)    
    output_delta_l = Dense(1)(drop2)
    output_delta_theta = Dense(1)(drop2)
    output_delta_psi = Dense(1)(drop2)

    model = Model(inputs = input_gyro_acc, outputs = [output_delta_l, output_delta_theta, output_delta_psi])
    model.summary()
    model.compile(optimizer = Adam(0.0001), loss = 'mean_squared_error')
    
    return model


def create_model_2d(window_size=200):
    input_gyro_acc = Input((window_size, 6))
    lstm1 = Bidirectional(CuDNNLSTM(128, return_sequences=True))(input_gyro_acc)    
    drop1 = Dropout(0.25)(lstm1)
    lstm2 = Bidirectional(CuDNNLSTM(128))(drop1)
    drop2 = Dropout(0.25)(lstm2)    
    output_delta_l = Dense(1)(drop2)
    output_delta_psi = Dense(1)(drop2)
    model = Model(inputs = input_gyro_acc, outputs = [output_delta_l, output_delta_psi])
    model.summary()
    model.compile(optimizer = Adam(0.0001), loss = 'mean_squared_error')
    
    return model
