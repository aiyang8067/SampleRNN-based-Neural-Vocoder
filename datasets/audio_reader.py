"""
Conditional SampleRNN Model Based on TensorFlow 1.0.0

Author: yangai
Email: ay8067@mail.ustc.edu.cn

Function: Reading audios for simple GPU version

"""
import numpy as np
import threading
import librosa
import glob
import os
import tensorflow as tf

def ReadFloatRawMat(datafile,column):

    data = np.fromfile(datafile,dtype=np.float32)
    if len(data)%column!=0:
        print 'ReadFloatRawMat %s, column wrong!'%datafile
        exit()
    if len(data)==0:
        print 'empty file: %s'%datafile
        exit()
    data.shape = [len(data)/column,column]
    return np.float32(data)

def __round_to(x, y):

    return int(np.ceil(x / float(y))) * y

def __linear_quantize(data, q_levels):

    eps = np.float64(1e-5)
    data *= (q_levels - eps)
    data += eps/2
    data = data.astype('int32')
    return data

def linear2mu(x, q_levels):

    mu=q_levels-1
    x_mu = np.sign(x) * np.log(1 + mu*np.abs(x))/np.log(1 + mu)
    return ((x_mu + 1)/2 * mu).astype('int16')

def mu2linear(x, q_levels):

    mu = float(q_levels-1)
    x = x.astype('float32')
    y = 2. * (x - (mu+1.)/2.) / (mu+1.)
    return np.sign(y) * (1./mu) * ((1. + mu)**np.abs(y) - 1.)

def __mu_law_quantize(data,q_levels):

    return linear2mu(data,q_levels)

def __batch_quantize(data, q_levels, q_type):

    data = data.astype('float64')
    if q_type == 'linear':
        return __linear_quantize(data, q_levels)
    if q_type == 'mu-law':
        return __mu_law_quantize(data,q_levels)
    raise NotImplementedError

def generate_sorted_list(wav_path,con_path,sample_rate,sorted_wav_list_path,sorted_con_list_path):

    paths = sorted(glob.glob(wav_path+"/*.wav"))
    con_paths=os.listdir(con_path)
    wav_len=[]
    for i in xrange(len(paths)):
        audio, _ = librosa.load(paths[i], sr=sample_rate, mono=True)
        wav_len.append(len(audio))
    wav_len_and_index=zip(wav_len,range(len(wav_len)))
    wav_len_and_index.sort(key=lambda x:x[0],reverse=True)
    index=[x[1] for x in wav_len_and_index]
    fid_wav=open(sorted_wav_list_path,'w')
    fid_con=open(sorted_con_list_path,'w')
    for i in xrange(len(index)):
        fid_wav.write(paths[index[i]]+'\n')
        fid_con.write(con_path+paths[index[i]].split('/')[-1].split('.')[0]+'.'+con_paths[0].split()[0].split('.')[-1]+'\n')
    fid_wav.close()
    fid_con.close()

def make_batches(wav_sorted_list,con_sorted_list,batch_size,con_frame_size,con_dim,sample_rate):

    fid_wav=open(wav_sorted_list,'r')
    fid_con=open(con_sorted_list,'r')

    wav_sorted_list=fid_wav.readlines()
    con_sorted_list=fid_con.readlines()

    fid_wav.close()
    fid_con.close()

    wav_batches=[]
    mask_batches=[]
    con_batches=[]

    for i in xrange(len(wav_sorted_list) / batch_size+1):
        if i==len(wav_sorted_list) / batch_size:
            if len(wav_sorted_list)%batch_size==0:
                break
            else:
                index=range(len(wav_sorted_list)-len(wav_sorted_list)%batch_size,len(wav_sorted_list))
        else:
            index=range(i*batch_size,(i+1)*batch_size)

        for j in xrange(len(index)):
            audio, _ = librosa.load(wav_sorted_list[index[j]].split()[0], sr=sample_rate, mono=True)
            condition=ReadFloatRawMat(con_sorted_list[index[j]].split()[0],1).reshape(1,-1)
            if len(audio)>condition.shape[1]/con_dim*con_frame_size:
                diff=len(audio)-condition.shape[1]/con_dim*con_frame_size
                audio=audio[:-diff]
            if len(audio)<condition.shape[1]/con_dim*con_frame_size:
                diff=condition.shape[1]/con_dim*con_frame_size-len(audio)
                audio=audio[:-(con_frame_size-diff)]
                condition=condition[:,:-con_dim]

            if j==0:
                max_len=len(audio)
                max_con_len=condition.shape[1]
                audio_mat=np.array(audio,dtype='float32').reshape(1,len(audio))
                mask_mat=np.ones(audio_mat.shape,dtype='int32')
                con_mat=condition
            else:
                current_len=len(audio)
                current_con_len=condition.shape[1]
                audio_mat=np.concatenate((audio_mat,np.pad(np.array(audio,dtype='float32').reshape(1,current_len),[[0,0],[0,max_len-current_len]],'constant')),axis=0)
                mask_mat=np.concatenate((mask_mat,np.pad(np.ones((1,current_len),dtype='int32'),[[0,0],[0,max_len-current_len]],'constant')),axis=0)
                con_mat=np.concatenate((con_mat,np.pad(condition,[[0,0],[0,max_con_len-current_con_len]],'constant')),axis=0)

        wav_batches.append(audio_mat)
        mask_batches.append(mask_mat)
        con_batches.append(con_mat)

    return wav_batches,mask_batches,con_batches

def load_data(wav_sorted_list,con_sorted_list,batch_size,seq_len,con_frame_size,con_dim,q_levels,q_type,sample_rate):

    wav_batches,mask_batches,con_batches=make_batches(wav_sorted_list,con_sorted_list,batch_size,con_frame_size,con_dim,sample_rate)

    cumulative_end_batch=0

    for index,wav_batch in enumerate(wav_batches):

        batch_num=np.int32(len(wav_batch))
        mask_batch=mask_batches[index]
        con_batch=con_batches[index]

        batch_seq_len = len(wav_batch[0]) 
        batch_seq_len = __round_to(batch_seq_len, seq_len)

        wav_batch=np.pad(wav_batch,[[0,0],[0,batch_seq_len-wav_batch.shape[1]]],'constant')
        mask_batch=np.pad(mask_batch,[[0,0],[0,batch_seq_len-mask_batch.shape[1]]],'constant')
        con_batch=np.pad(con_batch,[[0,0],[0,batch_seq_len/con_frame_size*con_dim-con_batch.shape[1]]],'constant')

        wav_batch = __batch_quantize(wav_batch, q_levels, q_type)

        q_zero=np.int32((q_levels-1)//2)

        wav_batch = np.concatenate([
            np.full((batch_num, con_frame_size), q_zero, dtype='int32'),
            wav_batch
            ], axis=1)

        mask_batch = np.concatenate([
            np.full((batch_num, con_frame_size), 1, dtype='int32'),
            mask_batch
        ], axis=1)

        con_batch = np.concatenate([
            np.full((batch_num, con_dim), 0, dtype='float32'),
            con_batch
        ], axis=1)

        for i in xrange(batch_seq_len // seq_len):
            reset_batch = np.int32(i==0)
            end_batch=np.int32(i==batch_seq_len // seq_len-1)
            sub_wav_batch = wav_batch[:, i*seq_len : (i+1)*seq_len+con_frame_size]
            sub_mask_batch = mask_batch[:, i*seq_len : (i+1)*seq_len+con_frame_size]
            sub_con_batch = con_batch[:, i*seq_len/con_frame_size*con_dim : (i+1)*seq_len/con_frame_size*con_dim+con_dim]
            cumulative_end_batch+=end_batch
            end_epoch=cumulative_end_batch/len(wav_batches)
            yield (sub_wav_batch,sub_mask_batch,sub_con_batch,reset_batch,end_batch,end_epoch)

class AudioReader(object):

    def __init__(self,
                 wav_sorted_list,
                 con_sorted_list,
                 batch_size,
                 seq_len,
                 con_frame_size,
                 con_dim,
                 q_levels,
                 q_type,
                 sample_rate,
                 coord,
                 queue_size=1):

        self.wav_sorted_list = wav_sorted_list
        self.con_sorted_list = con_sorted_list
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.con_frame_size = con_frame_size
        self.con_dim = con_dim
        self.q_levels = q_levels
        self.q_type = q_type
        self.sample_rate = sample_rate
        self.coord = coord
        self.threads = []
        self.wav_placeholder = tf.placeholder(dtype=tf.int32, shape=None)
        self.mask_placeholder=tf.placeholder(dtype=tf.int32,shape=None)
        self.con_placeholder=tf.placeholder(dtype=tf.float32,shape=None)
        self.reset_batch_placeholder = tf.placeholder(dtype=tf.int32, shape=None)
        self.end_batch_placeholder = tf.placeholder(dtype=tf.int32, shape=None)
        self.end_epoch_placeholder = tf.placeholder(dtype=tf.int32, shape=None)
        self.queue = tf.PaddingFIFOQueue(queue_size,
                                         ['int32','int32','float32','int32','int32','int32'],
                                         shapes=[[None,self.seq_len+self.con_frame_size],[None,self.seq_len+self.con_frame_size],
                                         [None,self.seq_len/self.con_frame_size*self.con_dim+self.con_dim],[],[],[]])
        self.enqueue = self.queue.enqueue([self.wav_placeholder,self.mask_placeholder,self.con_placeholder,
                                          self.reset_batch_placeholder,self.end_batch_placeholder,self.end_epoch_placeholder])

    def dequeue(self):
        output = self.queue.dequeue()
        return output

    def thread_main(self, sess):
        stop = False
        # Go through the dataset multiple times

        while not stop:
            count_end_batch=0
            end_epoch=0
            iterator = load_data(self.wav_sorted_list,
                                 self.con_sorted_list,
                                 self.batch_size,
                                 self.seq_len,
                                 self.con_frame_size,
                                 self.con_dim,
                                 self.q_levels,
                                 self.q_type,
                                 self.sample_rate)
            for sub_wav_batch, sub_mask_batch, sub_con_batch, reset_batch, end_batch, end_epoch in iterator:
                if self.coord.should_stop():
                    stop = True
                    break

                sess.run(self.enqueue,
                         feed_dict={self.wav_placeholder: sub_wav_batch,
                                    self.mask_placeholder: sub_mask_batch,
                                    self.con_placeholder: sub_con_batch,
                                    self.reset_batch_placeholder: reset_batch,
                                    self.end_batch_placeholder: end_batch,
                                    self.end_epoch_placeholder: end_epoch})

    def start_threads(self, sess, n_threads=1):
        for _ in range(n_threads):
            thread = threading.Thread(target=self.thread_main, args=(sess,))
            thread.daemon = True  # Thread will close when parent quits.
            thread.start()
            self.threads.append(thread)
        return self.threads
