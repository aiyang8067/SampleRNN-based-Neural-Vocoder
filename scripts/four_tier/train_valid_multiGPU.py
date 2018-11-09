"""
Conditional SampleRNN Model Based on TensorFlow 1.0.0

Author: yangai
Email: ay8067@mail.ustc.edu.cn

Function: Training and validation for multiple GPU version

Usage:
python -u scripts/four_tier/train_valid_multiGPU.py --seq_len 480 --con_dim 43 --con_frame_size 80 --big_frame_size 8
--frame_size 2 --weight_norm True --emb_size 256 --rnn_hidden_dim [[1024],[1024],[1024]] --rnn_type GRU --dnn_hidden_dim [1024,1024]
--learn_h0 True --q_levels 256 --q_type mu-law --batch_size 50 --gpu_id_list [0,1,2,3] (--restore_from 'results_4t')

"""

import tensorflow as tf
import numpy as np

from time import time
from datetime import datetime

import os, sys
sys.path.insert(1, os.getcwd())
import argparse
import itertools

from datasets.audio_reader_multiGPU import generate_sorted_list,AudioReader
from lib.model import SampleRNNModel

TRAIN_WAV_PATH='./datasets/dataset/waveform/train/'
TRAIN_CON_PATH='./datasets/dataset/acoustic_condition/train/'
VALID_WAV_PATH='./datasets/dataset/waveform/valid/'
VALID_CON_PATH='./datasets/dataset/acoustic_condition/valid/'
SORTED_LIST_PATH='./datasets/sorted_list/'
TRAIN_WAV_SORTED_LIST=SORTED_LIST_PATH+'train_wav_sorted_list.scp'
TRAIN_CON_SORTED_LIST=SORTED_LIST_PATH+'train_con_sorted_list.scp'
VALID_WAV_SORTED_LIST=SORTED_LIST_PATH+'valid_wav_sorted_list.scp'
VALID_CON_SORTED_LIST=SORTED_LIST_PATH+'valid_con_sorted_list.scp'
MODEL_DIR='./results_4t'
LOG_FILE='./log.txt'
TIER_NUM=4
GRAD_CLIP=1
SAMPLE_RATE=16000
PRINT_STEPS=3000
STOP_EPOCHS=30
LRATE_REDUCE_EPOCH=19
LEARNING_RATE=0.001

def get_args():
    def t_or_f(arg):
        ua = str(arg).upper()
        if 'TRUE'.startswith(ua):
            return True
        elif 'FALSE'.startswith(ua):
            return False
        else:
           raise ValueError('Arg is neither `True` nor `False`')

    def check_non_negative(value):
        ivalue = int(value)
        if ivalue < 0:
             raise argparse.ArgumentTypeError("%s is not non-negative!" % value)
        return ivalue

    def check_positive(value):
        ivalue = int(value)
        if ivalue < 1:
             raise argparse.ArgumentTypeError("%s is not positive!" % value)
        return ivalue

    def check_unit_interval(value):
        fvalue = float(value)
        if fvalue < 0 or fvalue > 1:
             raise argparse.ArgumentTypeError("%s is not in [0, 1] interval!" % value)
        return fvalue

    def check_list_in_list(value):
        temp_list=[]
        value=value.split('],')
        for i in xrange(len(value)):
            if i==0:
                sub_int_temp_list=[]
                sub_temp_list=value[i][2:].split(',')
                for j in xrange(len(sub_temp_list)):
                    sub_int_temp_list.append(int(sub_temp_list[j]))
                temp_list.append(sub_int_temp_list)
            elif i==len(value)-1:
                sub_int_temp_list=[]
                sub_temp_list=value[i][1:-2].split(',')
                for j in xrange(len(sub_temp_list)):
                    sub_int_temp_list.append(int(sub_temp_list[j]))
                temp_list.append(sub_int_temp_list)
            else:
                sub_int_temp_list=[]
                sub_temp_list=value[i][1:].split(',')
                for j in xrange(len(sub_temp_list)):
                    sub_int_temp_list.append(int(sub_temp_list[j]))
                temp_list.append(sub_int_temp_list)
        return temp_list

    def check_list(value):
        temp_list=[]
        value=value.split('[')[-1].split(']')[0].split(',')
        for i in xrange(len(value)):
            temp_list.append(int(value[i]))
        return temp_list

    # No default value here. Indicate every single arguement.
    parser = argparse.ArgumentParser(
        description='train_valid.py\nNo default value! Indicate every argument.')

    # TODO: Fix the descriptions
    # Hyperparameter arguements:
    parser.add_argument('--seq_len', help='How many samples to include in each Truncated BPTT pass', type=check_positive, required=True)
    parser.add_argument('--con_dim', help='Condition dimension',\
            type=check_positive, required=True)
    parser.add_argument('--con_frame_size', help='How many samples per condition frame',\
            type=check_positive, required=True)
    parser.add_argument('--big_frame_size', help='How many samples per big frame',\
            type=check_positive, required=True)
    parser.add_argument('--frame_size', help='How many samples per frame',\
            type=check_positive, required=True)
    parser.add_argument('--weight_norm', help='Adding learnable weight normalization to all the linear layers (except for the embedding layer)',\
            type=t_or_f, required=True)
    parser.add_argument('--emb_size', help='Size of embedding layer (> 0)',
            type=check_positive, required=True)  # different than two_tier
    parser.add_argument('--rnn_hidden_dim', help='The hidden dim of RNN at tier 4,3,2',
            type=check_list_in_list, required=True)
    parser.add_argument('--rnn_type', help='GRU or LSTM', choices=['LSTM', 'GRU'],\
            required=True)
    parser.add_argument('--dnn_hidden_dim', help='The hidden dim of DNN at tier 1',
            type=check_list, required=True)
    parser.add_argument('--learn_h0', help='Whether to learn the initial state of RNN',\
            type=t_or_f, required=True)
    parser.add_argument('--q_levels', help='Number of bins for quantization of audio samples.',\
            type=check_positive, required=True)
    parser.add_argument('--q_type', help='Quantization in linear-scale or mu-law compandig.',\
            choices=['linear', 'mu-law'], required=True)
    parser.add_argument('--batch_size', help='size of mini-batch',
            type=check_positive, required=True)
    parser.add_argument('--gpu_id_list', help='A list of GPU ids',
            type=check_list, required=True)
    parser.add_argument('--restore_from', help='Directory in which to restore the model from.',\
            type=str,default=None)

    args = parser.parse_args()

    return args

def save_model(saver, sess, logdir, step):
    model_name = 'model.ckpt'
    checkpoint_path = os.path.join(logdir, model_name)
    print('Storing checkpoint to {} ...'.format(logdir))
    sys.stdout.flush()

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    saver.save(sess, checkpoint_path, global_step=step)
    print(' Done.')

def load_model(saver, sess, logdir):
    print("Trying to restore saved checkpoints from {} ...".format(logdir))

    ckpt = tf.train.get_checkpoint_state(logdir)
    if ckpt:
        print("  Checkpoint found: {}".format(ckpt.model_checkpoint_path))
        global_step = int(ckpt.model_checkpoint_path
                          .split('/')[-1]
                          .split('-')[-1])
        print("  Global step was: {}".format(global_step))
        print("  Restoring...")
        saver.restore(sess, ckpt.model_checkpoint_path)
        print(" Done.")
        return global_step
    else: 
        print(" No checkpoint found.")
        return None

def average_and_clip_gradients(nnet_grads,used_gpu_num):
    average_grads=[]

    for grads_and_tvars in zip(*nnet_grads):
        grads=[]
        for index,(g,_) in enumerate(grads_and_tvars):
            g=tf.where(tf.less(index,used_gpu_num),g,tf.zeros(shape=tf.shape(g),dtype=tf.float32))
            grads.append(g)
        average_grad=tf.div(sum(grads),tf.to_float(used_gpu_num))
        average_grad=tf.clip_by_value(average_grad,-GRAD_CLIP,GRAD_CLIP)

        v=grads_and_tvars[0][1]
        average_grad_and_var=(average_grad,v)
        average_grads.append(average_grad_and_var)

    return average_grads

def main():

    with tf.Graph().as_default(),tf.device('/cpu:0'):

        print "Experiment started at:", datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M')
        exp_start = time()

        args = get_args()

        SEQ_LEN=args.seq_len
        CON_DIM=args.con_dim
        CON_FRAME_SIZE=args.con_frame_size
        BIG_FRAME_SIZE=args.big_frame_size
        FRAME_SIZE=args.frame_size
        WEIGHT_NORM=args.weight_norm
        EMB_SIZE=args.emb_size
        RNN_HIDDEN_DIM=args.rnn_hidden_dim
        RNN_TYPE=args.rnn_type
        DNN_HIDDEN_DIM=args.dnn_hidden_dim
        LEARN_H0=args.learn_h0
        Q_LEVELS=args.q_levels
        Q_TYPE=args.q_type
        BATCH_SIZE=args.batch_size
        GPU_ID_LIST=args.gpu_id_list
        RESTORE_FROM=args.restore_from
        GPU_NUM=len(GPU_ID_LIST)
        H0_MULT=2 if RNN_TYPE=='LSTM' else 1

        assert SEQ_LEN % CON_FRAME_SIZE == 0,\
            'seq_len should be divisible by con_frame_size'
        assert CON_FRAME_SIZE % BIG_FRAME_SIZE == 0,\
            'con_frame_size should be divisible by big_frame_size'
        assert BIG_FRAME_SIZE % FRAME_SIZE == 0,\
            'big_frame_size should be divisible by frame_size'

        if os.path.exists(TRAIN_WAV_SORTED_LIST) and os.path.exists(TRAIN_CON_SORTED_LIST) \
            and os.path.exists(VALID_WAV_SORTED_LIST) and os.path.exists(VALID_CON_SORTED_LIST):
            print 'Train and valid sorted list already exist!'
        else:
            if not os.path.exists(SORTED_LIST_PATH):
                os.makedirs(SORTED_LIST_PATH)
            print 'generating train and valid sorted list...'
            generate_sorted_list(TRAIN_WAV_PATH,TRAIN_CON_PATH,SAMPLE_RATE,TRAIN_WAV_SORTED_LIST,TRAIN_CON_SORTED_LIST)
            generate_sorted_list(VALID_WAV_PATH,VALID_CON_PATH,SAMPLE_RATE,VALID_WAV_SORTED_LIST,VALID_CON_SORTED_LIST)
            print 'Done.'

        coord_train_list=[]
        coord_valid_list=[]
        for i in xrange(GPU_NUM):
            coord_train_list.append(tf.train.Coordinator())
            coord_valid_list.append(tf.train.Coordinator())

        with tf.name_scope('create_inputs'):
            reader_train_list=[]
            reader_valid_list=[]

            for i in xrange(GPU_NUM):
                reader_train=AudioReader(
                    TRAIN_WAV_SORTED_LIST,
                    TRAIN_CON_SORTED_LIST,
                    BATCH_SIZE,
                    SEQ_LEN,
                    CON_FRAME_SIZE,
                    CON_DIM,
                    Q_LEVELS,
                    Q_TYPE,
                    SAMPLE_RATE,
                    GPU_NUM,
                    i,
                    coord_train_list[i])

                reader_valid=AudioReader(
                    VALID_WAV_SORTED_LIST,
                    VALID_CON_SORTED_LIST,
                    BATCH_SIZE,
                    SEQ_LEN,
                    CON_FRAME_SIZE,
                    CON_DIM,
                    Q_LEVELS,
                    Q_TYPE,
                    SAMPLE_RATE,
                    GPU_NUM,
                    i,
                    coord_valid_list[i])

                reader_train_list.append(reader_train)
                reader_valid_list.append(reader_valid)

        wav_batch_placeholder=tf.placeholder(dtype=tf.int32,shape=[None,SEQ_LEN+CON_FRAME_SIZE])
        mask_batch_placeholder=tf.placeholder(dtype=tf.int32,shape=[None,SEQ_LEN+CON_FRAME_SIZE])
        con_batch_placeholder=tf.placeholder(dtype=tf.float32,shape=[None,SEQ_LEN/CON_FRAME_SIZE*CON_DIM+CON_DIM])
        con_h0_placeholder=tf.placeholder(dtype=tf.float32,shape=[None,H0_MULT*sum(RNN_HIDDEN_DIM[0])])
        big_h0_placeholder=tf.placeholder(dtype=tf.float32,shape=[None,H0_MULT*sum(RNN_HIDDEN_DIM[1])])
        h0_placeholder=tf.placeholder(dtype=tf.float32,shape=[None,H0_MULT*sum(RNN_HIDDEN_DIM[2])])
        reset_batch_placeholder=tf.placeholder(dtype=tf.int32,shape=[])
        end_batch_placeholder=tf.placeholder(dtype=tf.int32,shape=[])
        end_epoch_placeholder=tf.placeholder(dtype=tf.int32,shape=[])

        all_h0_train_list=[]
        all_h0_valid_list=[]
        for i in xrange(GPU_NUM):
            temp_train_list=[]
            temp_valid_list=[]
            for j in xrange(TIER_NUM-1):
                temp_train_list.append(tf.placeholder(dtype=tf.float32,shape=[None,H0_MULT*sum(RNN_HIDDEN_DIM[j])]))
                temp_valid_list.append(tf.placeholder(dtype=tf.float32,shape=[None,H0_MULT*sum(RNN_HIDDEN_DIM[j])]))
            all_h0_train_list.append(temp_train_list)
            all_h0_valid_list.append(temp_valid_list)

        with tf.variable_scope('SampleRNNModel',reuse=None):

            net=SampleRNNModel(
                seq_len=SEQ_LEN,
                con_dim=CON_DIM,
                con_frame_size=CON_FRAME_SIZE,
                big_frame_size=BIG_FRAME_SIZE,
                frame_size=FRAME_SIZE,
                weight_norm=WEIGHT_NORM,
                emb_size=EMB_SIZE,
                rnn_hidden_dim=RNN_HIDDEN_DIM,
                dnn_hidden_dim=DNN_HIDDEN_DIM,
                rnn_type=RNN_TYPE,
                learn_h0=LEARN_H0,
                q_levels=Q_LEVELS)

            _,_,_,_,_,_,_=net.loss_and_info_for_train_mutiGPU(
                wav_batch_placeholder,mask_batch_placeholder,con_batch_placeholder,
                con_h0_placeholder,big_h0_placeholder,h0_placeholder,reset_batch_placeholder,
                end_batch_placeholder,end_epoch_placeholder)

        lrate=tf.Variable(0.0,trainable=False)
        new_lrate=tf.placeholder(dtype=tf.float32,shape=[])
        lr_update=tf.assign(lrate,new_lrate)

        optimizer=tf.train.AdamOptimizer(learning_rate=lrate)

        tvars=tf.trainable_variables()
        print 'All parameters:'
        for i in xrange(len(tvars)):
            print tvars[i].name
            print tvars[i].get_shape()

        loss_and_info_list_train=[[],[],[],[],[],[],[]]
        wait_flag_list_train=[]
        grads_list=[]

        with tf.variable_scope('SampleRNNModel',reuse=True):

            for i in xrange(GPU_NUM):
                with tf.device('/gpu:%d' % GPU_ID_LIST[i]):
                    with tf.name_scope('train_%d' % GPU_ID_LIST[i]):

                        wav_batch_train,mask_batch_train,con_batch_train,reset_batch_train,\
                            end_batch_train,end_epoch_train=reader_train_list[i].dequeue()

                        loss_and_info_for_train=net.loss_and_info_for_train_mutiGPU(
                                wav_batch_train,mask_batch_train,con_batch_train,all_h0_train_list[i][0],
                                all_h0_train_list[i][1],all_h0_train_list[i][2],reset_batch_train,end_batch_train,
                                end_epoch_train)

                        wait_flag_list_train.append(
                            tf.where(tf.cast(loss_and_info_for_train[1],'bool'),
                                tf.constant(1,dtype=tf.int32),tf.constant(0,dtype=tf.int32)))

                        ce_loss_average=tf.where(tf.cast(loss_and_info_for_train[1],'bool'),
                            tf.div(loss_and_info_for_train[0],loss_and_info_for_train[1]),
                            loss_and_info_for_train[0])

                        grads=optimizer.compute_gradients(ce_loss_average,var_list=tvars)
                        grads_list.append(grads)

                        for j in xrange(len(loss_and_info_list_train)):
                            loss_and_info_list_train[j].append(loss_and_info_for_train[j])

        average_grads_and_tvars=average_and_clip_gradients(grads_list,tf.reduce_sum(wait_flag_list_train,0))
        train_op=optimizer.apply_gradients(average_grads_and_tvars)

        loss_and_info_list_valid=[[],[],[],[],[],[],[],[],[]]

        with tf.variable_scope('SampleRNNModel',reuse=True):

            for i in xrange(GPU_NUM):
                with tf.device('/gpu:%d' % GPU_ID_LIST[i]):
                    with tf.name_scope('valid_%d' % GPU_ID_LIST[i]):

                        wav_batch_valid,mask_batch_valid,con_batch_valid,reset_batch_valid,\
                            end_batch_valid,end_epoch_valid=reader_valid_list[i].dequeue()

                        loss_and_info_for_valid=net.loss_and_info_for_valid(
                                wav_batch_valid,mask_batch_valid,con_batch_valid,all_h0_valid_list[i][0],
                                all_h0_valid_list[i][1],all_h0_valid_list[i][2],reset_batch_valid,end_batch_valid,
                                end_epoch_valid)

                        for j in xrange(len(loss_and_info_for_valid)):
                            loss_and_info_list_valid[j].append(loss_and_info_for_valid[j])

        sess=tf.Session(config=tf.ConfigProto(log_device_placement=False,allow_soft_placement=True))
        init=tf.global_variables_initializer()
        sess.run(init)

        saver=tf.train.Saver(var_list=tvars,max_to_keep=400)

        if RESTORE_FROM is None:
            global_step=0
        else:
            global_step=load_model(saver, sess, RESTORE_FROM)

        threads_train_list=[]
        threads_valid_list=[]
        for i in xrange(GPU_NUM):
            threads_train=tf.train.start_queue_runners(sess=sess,coord=coord_train_list[i])
            reader_train_list[i].start_threads(sess)
            threads_train_list.append(threads_train)

            threads_valid=tf.train.start_queue_runners(sess=sess,coord=coord_valid_list[i])
            reader_valid_list[i].start_threads(sess)
            threads_valid_list.append(threads_valid)

        fid_log=open(LOG_FILE,'w')    

        try:
            train_total_time=0.
            epoch=1
            last_print_step=global_step
            lowest_valid_loss=np.finfo(np.float32).max
            ce_loss_train=[]

            NEW_LRATE=LEARNING_RATE

            feed_dict_train=dict()
            feed_dict_valid=dict()
            for i in xrange(GPU_NUM):
                for j in xrange(TIER_NUM-1):
                    feed_dict_train[all_h0_train_list[i][j]]=np.zeros((BATCH_SIZE,H0_MULT*sum(RNN_HIDDEN_DIM[j])))
                    feed_dict_valid[all_h0_valid_list[i][j]]=np.zeros((BATCH_SIZE,H0_MULT*sum(RNN_HIDDEN_DIM[j])))

            print 'Training!'
            print 'Now learning rate is {}.'.format(LEARNING_RATE)
            while True:
                if global_step%500==0:
                    print global_step,

                global_step+=1

                sess.run(lr_update,feed_dict={new_lrate:NEW_LRATE})

                train_start_time=time()
                loss_and_info_list_train_value,_=sess.run([loss_and_info_list_train,train_op],feed_dict=feed_dict_train)
                ce_loss_sum_train_list,target_mask_sum_train_list,NEW_CON_H0_TRAIN_LIST,NEW_BIG_H0_TRAIN_LIST,\
                    NEW_H0_TRAIN_LIST,END_BATCH_TRAIN_LIST,END_EPOCH_TRAIN_LIST=loss_and_info_list_train_value
                NEW_ALL_H0_TRAIN_LIST=[NEW_CON_H0_TRAIN_LIST,NEW_BIG_H0_TRAIN_LIST,NEW_H0_TRAIN_LIST]

                train_total_time+=time()-train_start_time
                ce_loss_train.append(sum(ce_loss_sum_train_list)/sum(target_mask_sum_train_list))

                for i in xrange(GPU_NUM):
                    for j in xrange(TIER_NUM-1):
                        if END_BATCH_TRAIN_LIST[i]:
                            feed_dict_train[all_h0_train_list[i][j]]=np.zeros((BATCH_SIZE,H0_MULT*sum(RNN_HIDDEN_DIM[j])))
                        else:
                            feed_dict_train[all_h0_train_list[i][j]]=NEW_ALL_H0_TRAIN_LIST[j][i]

                if global_step-last_print_step==PRINT_STEPS//GPU_NUM or sum(END_EPOCH_TRAIN_LIST)==GPU_NUM:

                    print '\nValidation!'
                    ce_loss_list_valid=[]
                    accuracy_list_valid=[]
                    ce_loss_batch_list_valid=[]
                    accuracy_batch_list_valid=[]
                    target_mask_batch_list_valid=[]

                    for i in xrange(GPU_NUM):
                        ce_loss_batch_list_valid.append([])
                        accuracy_batch_list_valid.append([])
                        target_mask_batch_list_valid.append([])

                    valid_start_time=time()
                    while True:
                        loss_and_info_list_valid_value=sess.run(loss_and_info_list_valid,feed_dict=feed_dict_valid)
                        ce_loss_batch_sum_valid_list,accuracy_batch_sum_valid_list,target_mask_batch_sum_valid_list,CON_H0_VALID_LIST,\
                            BIG_H0_VALID_LIST,H0_VALID_LIST,RESET_BATCH_VALID_LIST,END_BATCH_VALID_LIST,END_EPOCH_VALID_LIST=loss_and_info_list_valid_value
                        NEW_ALL_H0_VALID_LIST=[CON_H0_VALID_LIST,BIG_H0_VALID_LIST,H0_VALID_LIST]

                        for i in xrange(GPU_NUM):
                            if RESET_BATCH_VALID_LIST[i]:
                                ce_loss_batch_list_valid[i]=[]
                                accuracy_batch_list_valid[i]=[]
                                target_mask_batch_list_valid[i]=[]

                            ce_loss_batch_list_valid[i].append(ce_loss_batch_sum_valid_list[i])
                            accuracy_batch_list_valid[i].append(accuracy_batch_sum_valid_list[i])
                            target_mask_batch_list_valid[i].append(target_mask_batch_sum_valid_list[i])

                            for j in xrange(TIER_NUM-1):
                                if END_BATCH_VALID_LIST[i]:
                                    feed_dict_valid[all_h0_valid_list[i][j]]=np.zeros((BATCH_SIZE,H0_MULT*sum(RNN_HIDDEN_DIM[j])))
                                else:
                                    feed_dict_valid[all_h0_valid_list[i][j]]=NEW_ALL_H0_VALID_LIST[j][i]

                            if END_BATCH_VALID_LIST[i] and np.sum(sum(target_mask_batch_list_valid[i]))!=0:
                                ce_loss_list_valid.extend(list(sum(ce_loss_batch_list_valid[i])/sum(target_mask_batch_list_valid[i])))
                                accuracy_list_valid.extend(list(sum(accuracy_batch_list_valid[i])/sum(target_mask_batch_list_valid[i])))

                        if sum(END_EPOCH_VALID_LIST)==GPU_NUM:
                            ce_loss_valid=np.mean(ce_loss_list_valid)
                            accuracy_valid=np.mean(accuracy_list_valid)
                            break

                    valid_time=time()-valid_start_time
                    if ce_loss_valid<lowest_valid_loss:
                        lowest_valid_loss=ce_loss_valid
                        print "\n>>> Best validation loss of {} reached.".format(lowest_valid_loss)

                    save_model(saver, sess, MODEL_DIR, global_step)

                    print_info = "epoch:{}\ttotal steps:{}\twall clock time:{:.2f}h\n"
                    print_info += ">>> Lowest valid cost:{}\n"
                    print_info += "\ttrain loss:{:.4f}\ttotal time:{:.2f}h\tper step:{:.3f}s\n"
                    print_info += "\tvalid loss:{:.4f}\tvalid accuracy:{:.4f}%\ttotal time:{:.2f}h\n"
                    print_info = print_info.format(epoch,
                                                   global_step,
                                                   (time()-exp_start)/3600,
                                                   lowest_valid_loss,
                                                   np.mean(ce_loss_train),
                                                   train_total_time/3600,
                                                   train_total_time/global_step,
                                                   ce_loss_valid,
                                                   accuracy_valid*100,
                                                   valid_time/3600)
                    print print_info

                    fid_log.write(print_info)
                    fid_log.flush()

                    print "Validation Done!\nBack to Training..."

                    if global_step-last_print_step==PRINT_STEPS//GPU_NUM:
                        ce_loss_train=[]
                        last_print_step+=PRINT_STEPS//GPU_NUM

                    if sum(END_EPOCH_TRAIN_LIST)==GPU_NUM:
                        if epoch==STOP_EPOCHS:
                            print "Experiment ended at:", datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M')
                            print "Wall clock time spent: {:.2f}h".format((time()-exp_start)/3600)
                            break
                            
                        print "[Another epoch]"
                        epoch+=1
                        if epoch==LRATE_REDUCE_EPOCH:
                            NEW_LRATE=LEARNING_RATE*0.1
                            print "From now on, the learning rate is {}.".format(NEW_LRATE)

        except KeyboardInterrupt:
            # Introduce a line break after ^C is displayed so save message
            # is on its own line.
            print()

        finally:
            for i in xrange(GPU_NUM):
                coord_train_list[i].request_stop()
                coord_train_list[i].join(threads_train_list[i])
                coord_valid_list[i].request_stop()
                coord_valid_list[i].join(threads_valid_list[i])
            fid_log.close()

if __name__=='__main__':
    main()