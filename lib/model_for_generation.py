"""
Conditional SampleRNN Model Based on TensorFlow 1.0.0

Author: yangai
Email: ay8067@mail.ustc.edu.cn

Function: Building the model

"""
import tensorflow as tf
import numpy as np
from lib.core_rnn_cell_impl import GRUCell,BasicLSTMCell,MultiRNNCell

def uniform(stdev, size):
    """
    uniform distribution with the given stdev and size

    """
    return np.random.uniform(
        low=-stdev * np.sqrt(3),
        high=stdev * np.sqrt(3),
        size=size
    ).astype('float32')

def create_zeros_variable(shape,name):

    zeros_value=np.zeros(shape,dtype='float32')

    zeros_variable=tf.Variable(zeros_value,name=name)

    return zeros_variable

def create_randn_variable(shape_row,shape_col,name):

    randn_value=np.random.randn(shape_row,shape_col).astype('float32')

    randn_variable=tf.Variable(randn_value,name=name)

    return randn_variable

def create_weight_variable(input_dim, output_dim, initialization=None, weightnorm=True):

    if initialization == 'lecun' or (initialization == None and input_dim != output_dim):
        weight_value = uniform(np.sqrt(1. / input_dim), (input_dim, output_dim))
    elif initialization == 'glorot':
        weight_value = uniform(np.sqrt(2./(input_dim+output_dim)), (input_dim, output_dim))
    elif initialization == 'he':
        weight_value = uniform(np.sqrt(2. / input_dim), (input_dim, output_dim))
    elif initialization == 'glorot_he':
        weight_value = uniform(np.sqrt(4./(input_dim+output_dim)), (input_dim, output_dim))
    elif initialization == 'orthogonal' or (initialization == None and input_dim == output_dim):
        # From lasagne
        def sample(shape):
            if len(shape) < 2:
                raise RuntimeError("Only shapes of length 2 or more are supported.")
            flat_shape = (shape[0], np.prod(shape[1:]))
            a = np.random.normal(0.0, 1.0, flat_shape)
            u, _, v = np.linalg.svd(a, full_matrices=False)
            q = u if u.shape == flat_shape else v
            q = q.reshape(shape)
            return q.astype('float32')
        weight_value = sample((input_dim, output_dim))
    else:
        raise Exception("Invalid initialization ({})!"\
                .format(repr(initialization)))

    weight=tf.Variable(weight_value,name='weight')
    norm=None

    if weightnorm:

        norm_value=np.linalg.norm(weight_value, axis=0)
        norm=tf.Variable(norm_value,name='norm')

    return weight, norm

def create_bias_variable(dim):

    bias_value=np.zeros((dim,),dtype='float32')
    bias=tf.Variable(bias_value,name='bias')

    return bias

def Linear(input, weight, bias=None, norm=None,dimension_dismatch=True):

    if norm is None:
        prepared_weight=weight
    else:
        prepared_weight=tf.multiply(weight,tf.div(norm,tf.sqrt(tf.reduce_sum(tf.multiply(weight,weight),0))))

    if dimension_dismatch:
        output=tf.einsum('ijk,kl->ijl',input,prepared_weight)
    else:
        output=tf.matmul(input,prepared_weight)

    if bias is not None:
        output=tf.add(output,bias)

    return output

def slide_window(matrix,window_size):

    for i in xrange(window_size):
        if i!=window_size-1:
            sub_mat=matrix[:,i:-(window_size-i-1)]
        else:
            sub_mat=matrix[:,i:]

        if i==0:
            mat=tf.reshape(sub_mat,[-1,1])
        else:
            mat=tf.concat([mat,tf.reshape(sub_mat,[-1,1])],1)

    return mat

class SampleRNNModel(object):
    '''Implements the SampleRNN network for generative audio.

    '''
    def __init__(self,
                 con_dim,
                 con_frame_size,
                 big_frame_size,
                 frame_size,
                 weight_norm,
                 emb_size,
                 rnn_hidden_dim,
                 dnn_hidden_dim,
                 rnn_type,
                 learn_h0,
                 q_levels):

        self.con_dim=con_dim
        self.con_frame_size=con_frame_size
        self.big_frame_size=big_frame_size
        self.frame_size=frame_size
        self.weight_norm=weight_norm
        self.emb_size=emb_size
        self.rnn_hidden_dim=rnn_hidden_dim
        self.dnn_hidden_dim=dnn_hidden_dim
        self.rnn_type=rnn_type
        self.learn_h0=learn_h0
        self.q_levels=q_levels
        self.h0_mult=2 if rnn_type=='LSTM' else 1

        self.variables = self.create_variables()

    def create_variables(self):

        var=dict()

        with tf.variable_scope('ConFrameLevel'):
            ConFrameVar=dict()

            with tf.variable_scope('Output'):
                weight, norm=create_weight_variable(self.rnn_hidden_dim[0][-1],self.rnn_hidden_dim[0][-1]*self.con_frame_size/self.big_frame_size,
                                                    initialization='he',weightnorm=self.weight_norm)

                ConFrameVar['output_weight']=weight
                if self.weight_norm:
                    ConFrameVar['output_norm']=norm

                ConFrameVar['output_bias']=create_bias_variable(self.rnn_hidden_dim[0][-1]*self.con_frame_size/self.big_frame_size)

            if self.learn_h0:
                ConFrameVar['h0']=create_zeros_variable((1,self.h0_mult*sum(self.rnn_hidden_dim[0])),name='h0')

            var['ConFrameLevel']=ConFrameVar

        with tf.variable_scope('BigFrameLevel'):
            BigFrameVar=dict()

            with tf.variable_scope('Output'):
                weight, norm=create_weight_variable(self.rnn_hidden_dim[1][-1],self.rnn_hidden_dim[1][-1]*self.big_frame_size/self.frame_size,
                                                    initialization='he',weightnorm=self.weight_norm)

                BigFrameVar['output_weight']=weight
                if self.weight_norm:
                    BigFrameVar['output_norm']=norm

                BigFrameVar['output_bias']=create_bias_variable(self.rnn_hidden_dim[1][-1]*self.big_frame_size/self.frame_size)

            with tf.variable_scope('InputExpand'):
                weight, norm=create_weight_variable(self.big_frame_size,self.rnn_hidden_dim[0][-1],
                                                initialization='he',weightnorm=self.weight_norm)

                BigFrameVar['inputexpand_weight']=weight
                if self.weight_norm:
                    BigFrameVar['inputexpand_norm']=norm

                BigFrameVar['inputexpand_bias']=create_bias_variable(self.rnn_hidden_dim[0][-1])

            if self.learn_h0:
                BigFrameVar['h0']=create_zeros_variable((1,self.h0_mult*sum(self.rnn_hidden_dim[1])),name='h0')

            var['BigFrameLevel']=BigFrameVar

        with tf.variable_scope('FrameLevel'):
            FrameVar=dict()

            with tf.variable_scope('Output'):
                weight, norm=create_weight_variable(self.rnn_hidden_dim[2][-1],self.rnn_hidden_dim[2][-1]*self.frame_size,
                                                    initialization='he',weightnorm=self.weight_norm)

                FrameVar['output_weight']=weight
                if self.weight_norm:
                    FrameVar['output_norm']=norm

                FrameVar['output_bias']=create_bias_variable(self.rnn_hidden_dim[2][-1]*self.frame_size)

            with tf.variable_scope('InputExpand'):
                weight, norm=create_weight_variable(self.frame_size,self.rnn_hidden_dim[1][-1],
                                                    initialization='he',weightnorm=self.weight_norm)

                FrameVar['inputexpand_weight']=weight
                if self.weight_norm:
                    FrameVar['inputexpand_norm']=norm

                FrameVar['inputexpand_bias']=create_bias_variable(self.rnn_hidden_dim[1][-1])

            if self.learn_h0:
                FrameVar['h0']=create_zeros_variable((1,self.h0_mult*sum(self.rnn_hidden_dim[2])),name='h0')

            var['FrameLevel']=FrameVar

        with tf.variable_scope('SampleLevel'):
            SampleVar=dict()

            SampleVar['embedding']=create_randn_variable(self.q_levels,self.emb_size,name='embedding')

            with tf.variable_scope('InputExpand'):
                weight, norm=create_weight_variable(self.frame_size*self.emb_size,self.rnn_hidden_dim[2][-1],
                                                    initialization='he',weightnorm=self.weight_norm)

                SampleVar['inputexpand_weight']=weight
                if self.weight_norm:
                    SampleVar['inputexpand_norm']=norm

            for i in xrange(len(self.dnn_hidden_dim)):
                with tf.variable_scope('Hidden'+str(i)):
                    if i==0:
                        weight, norm=create_weight_variable(self.rnn_hidden_dim[2][-1],self.dnn_hidden_dim[i],
                                                            initialization='he',weightnorm=self.weight_norm)
                    else:
                        weight, norm=create_weight_variable(self.dnn_hidden_dim[i-1],self.dnn_hidden_dim[i],
                                                            initialization='he',weightnorm=self.weight_norm)

                    SampleVar['weight'+str(i)]=weight
                    if self.weight_norm:
                        SampleVar['norm'+str(i)]=norm

                    SampleVar['bias'+str(i)]=create_bias_variable(self.dnn_hidden_dim[i])

            with tf.variable_scope('Output'):
                weight, norm=create_weight_variable(self.dnn_hidden_dim[-1],self.q_levels,
                                                    initialization='he',weightnorm=self.weight_norm)

                SampleVar['output_weight']=weight
                if self.weight_norm:
                    SampleVar['output_norm']=norm

                SampleVar['output_bias']=create_bias_variable(self.q_levels)

            var['SampleLevel']=SampleVar

        return var


    def con_frame_level_rnn(self,con_batch, h0, reset_batch):

        frames=tf.reshape(con_batch,[tf.shape(con_batch)[0],-1,self.con_dim])

        h0=tf.slice(h0,[0,0],[tf.shape(con_batch)[0],-1])

        if self.learn_h0:
            h0=tf.where(tf.cast(reset_batch,'bool'),tf.tile(self.variables['ConFrameLevel']['h0'],[tf.shape(con_batch)[0],1]),h0)

        with tf.variable_scope('ConFrameLevelRNN'):

            if self.rnn_type=='GRU':
                cell=[]
                for i in xrange(len(self.rnn_hidden_dim[0])):
                    cell.append(GRUCell(num_units=self.rnn_hidden_dim[0][i],weight_norm=self.weight_norm))
                multi_cell=MultiRNNCell(cell,state_is_tuple=False)

                rnn_out,last_state=tf.nn.dynamic_rnn(multi_cell,frames,initial_state=h0,time_major=False)

            elif self.rnn_type=='LSTM':
                cell=[]
                for i in xrange(len(self.rnn_hidden_dim[0])):
                    cell.append(BasicLSTMCell(num_units=self.rnn_hidden_dim[0][i],state_is_tuple=False,weight_norm=self.weight_norm))
                multi_cell=MultiRNNCell(cell,state_is_tuple=False)

                rnn_out,last_state=tf.nn.dynamic_rnn(multi_cell,frames,initial_state=h0,time_major=False)

        if self.weight_norm:
            output=Linear(rnn_out,self.variables['ConFrameLevel']['output_weight'],
                                  self.variables['ConFrameLevel']['output_bias'],
                                  self.variables['ConFrameLevel']['output_norm'])
        else:
            output=Linear(rnn_out,self.variables['ConFrameLevel']['output_weight'],
                                  self.variables['ConFrameLevel']['output_bias'])

        output=tf.reshape(output,[tf.shape(output)[0],-1,self.rnn_hidden_dim[0][-1]])

        return output,last_state

    def big_frame_level_rnn(self,wav_batch, other_input, h0, reset_batch):

        frames=tf.reshape(wav_batch,[tf.shape(wav_batch)[0],-1,self.big_frame_size])

        frames=tf.to_float(frames)/(self.q_levels/2)-1

        if self.weight_norm:
            expand_inputs=tf.add(Linear(frames,self.variables['BigFrameLevel']['inputexpand_weight'],
                                               self.variables['BigFrameLevel']['inputexpand_bias'],
                                               self.variables['BigFrameLevel']['inputexpand_norm']),other_input)
        else:
            expand_inputs=tf.add(Linear(frames,self.variables['BigFrameLevel']['inputexpand_weight'],
                                               self.variables['BigFrameLevel']['inputexpand_bias']),other_input)

        h0=tf.slice(h0,[0,0],[tf.shape(wav_batch)[0],-1])

        if self.learn_h0:
            h0=tf.where(tf.cast(reset_batch,'bool'),tf.tile(self.variables['BigFrameLevel']['h0'],[tf.shape(wav_batch)[0],1]),h0)

        with tf.variable_scope('BigFrameLevelRNN'):

            if self.rnn_type=='GRU':
                cell=[]
                for i in xrange(len(self.rnn_hidden_dim[1])):
                    cell.append(GRUCell(num_units=self.rnn_hidden_dim[1][i],weight_norm=self.weight_norm))
                multi_cell=MultiRNNCell(cell,state_is_tuple=False)

                rnn_out,last_state=tf.nn.dynamic_rnn(multi_cell,expand_inputs,initial_state=h0,time_major=False)

            elif self.rnn_type=='LSTM':
                cell=[]
                for i in xrange(len(self.rnn_hidden_dim[1])):
                    cell.append(BasicLSTMCell(num_units=self.rnn_hidden_dim[1][i],state_is_tuple=False,weight_norm=self.weight_norm))
                multi_cell=MultiRNNCell(cell,state_is_tuple=False)

                rnn_out,last_state=tf.nn.dynamic_rnn(multi_cell,expand_inputs,initial_state=h0,time_major=False)

        if self.weight_norm:
            output=Linear(rnn_out,self.variables['BigFrameLevel']['output_weight'],
                                  self.variables['BigFrameLevel']['output_bias'],
                                  self.variables['BigFrameLevel']['output_norm'])
        else:
            output=Linear(rnn_out,self.variables['BigFrameLevel']['output_weight'],
                                  self.variables['BigFrameLevel']['output_bias'])

        output=tf.reshape(output,[tf.shape(output)[0],-1,self.rnn_hidden_dim[1][-1]])

        return output,last_state

    def frame_level_rnn(self,wav_batch, other_input, h0, reset_batch):

        frames=tf.reshape(wav_batch,[tf.shape(wav_batch)[0],-1,self.frame_size])

        frames=tf.to_float(frames)/(self.q_levels/2)-1

        if self.weight_norm:
            expand_inputs=tf.add(Linear(frames,self.variables['FrameLevel']['inputexpand_weight'],
                                               self.variables['FrameLevel']['inputexpand_bias'],
                                               self.variables['FrameLevel']['inputexpand_norm']),other_input)
        else:
            expand_inputs=tf.add(Linear(frames,self.variables['FrameLevel']['inputexpand_weight'],
                                               self.variables['FrameLevel']['inputexpand_bias']),other_input)

        h0=tf.slice(h0,[0,0],[tf.shape(wav_batch)[0],-1])

        if self.learn_h0:
            h0=tf.where(tf.cast(reset_batch,'bool'),tf.tile(self.variables['FrameLevel']['h0'],[tf.shape(wav_batch)[0],1]),h0)

        with tf.variable_scope('FrameLevelRNN'):

            if self.rnn_type=='GRU':
                cell=[]
                for i in xrange(len(self.rnn_hidden_dim[2])):
                    cell.append(GRUCell(num_units=self.rnn_hidden_dim[2][i],weight_norm=self.weight_norm))
                multi_cell=MultiRNNCell(cell,state_is_tuple=False)

                rnn_out,last_state=tf.nn.dynamic_rnn(multi_cell,expand_inputs,initial_state=h0,time_major=False)

            elif self.rnn_type=='LSTM':
                cell=[]
                for i in xrange(len(self.rnn_hidden_dim[2])):
                    cell.append(BasicLSTMCell(num_units=self.rnn_hidden_dim[2][i],state_is_tuple=False,weight_norm=self.weight_norm))
                multi_cell=MultiRNNCell(cell,state_is_tuple=False)

                rnn_out,last_state=tf.nn.dynamic_rnn(multi_cell,expand_inputs,initial_state=h0,time_major=False)

        if self.weight_norm:
            output=Linear(rnn_out,self.variables['FrameLevel']['output_weight'],
                                  self.variables['FrameLevel']['output_bias'],
                                  self.variables['FrameLevel']['output_norm'])
        else:
            output=Linear(rnn_out,self.variables['FrameLevel']['output_weight'],
                                  self.variables['FrameLevel']['output_bias'])

        output=tf.reshape(output,[tf.shape(output)[0],-1,self.rnn_hidden_dim[2][-1]])

        return output,last_state


    def sample_level_predictor(self,frame_level_outputs, prev_samples, is_argmax):

        prev_samples=tf.one_hot(prev_samples,depth=self.q_levels,dtype=tf.float32)
        prev_samples=Linear(prev_samples,self.variables['SampleLevel']['embedding'])
        prev_samples=tf.reshape(prev_samples,[tf.shape(prev_samples)[0],-1])

        if self.weight_norm:
            expand_inputs=tf.add(Linear(prev_samples,self.variables['SampleLevel']['inputexpand_weight'],
                                                     bias=None,
                                                     norm=self.variables['SampleLevel']['inputexpand_norm'],
                                                     dimension_dismatch=False),frame_level_outputs)
        else:
            expand_inputs=tf.add(Linear(prev_samples,self.variables['SampleLevel']['inputexpand_weight'],
                                               bias=None,norm=None,dimension_dismatch=False),frame_level_outputs)

        output=expand_inputs
        for i in xrange(len(self.dnn_hidden_dim)):

            if self.weight_norm:
                output=tf.nn.relu(Linear(output,self.variables['SampleLevel']['weight'+str(i)],
                                                self.variables['SampleLevel']['bias'+str(i)],
                                                self.variables['SampleLevel']['norm'+str(i)],
                                                dimension_dismatch=False))
            else:
                output=tf.nn.relu(Linear(output,self.variables['SampleLevel']['weight'+str(i)],
                                                self.variables['SampleLevel']['bias'+str(i)],
                                                norm=None,
                                                dimension_dismatch=False))

        if self.weight_norm:
            output=Linear(output,self.variables['SampleLevel']['output_weight'],
                                self.variables['SampleLevel']['output_bias'],
                                self.variables['SampleLevel']['output_norm'],
                                dimension_dismatch=False)
        else:
            output=Linear(output,self.variables['SampleLevel']['output_weight'],
                                self.variables['SampleLevel']['output_bias'],
                                norm=None,
                                dimension_dismatch=False)

        probability_distribution=tf.nn.softmax(output)

        new_sample=tf.where(tf.cast(is_argmax,'bool'),
            tf.reshape(tf.to_int32(tf.argmax(probability_distribution,1)),[-1,1]),
            tf.to_int32(tf.multinomial(tf.log(probability_distribution),num_samples=1)))

        return new_sample