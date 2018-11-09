"""
Conditional SampleRNN Model Based on TensorFlow 1.0.0

Author: yangai
Email: ay8067@mail.ustc.edu.cn

Function: Generator

Usage:
CUDA_VISIBLE_DEVICES=0 python -u scripts/four_tier/generation.py --seq_len 480 --con_dim 43 --con_frame_size 80 --big_frame_size 8
--frame_size 2 --weight_norm True --emb_size 256 --rnn_hidden_dim [[1024],[1024],[1024]] --rnn_type GRU --dnn_hidden_dim [1024,1024]
--learn_h0 True --q_levels 256 --q_type mu-law --batch_size 50 --wav_out_path 'wav_out' --restore_from 'results_4t'

"""

import tensorflow as tf
import numpy as np

from time import time
from datetime import datetime

import os, sys
sys.path.insert(1, os.getcwd())
import argparse
import itertools

import soundfile as sf

from datasets.audio_reader import generate_sorted_list,load_data
from lib.model import SampleRNNModel

TEST_WAV_PATH='./datasets/dataset/waveform/test/'
TEST_CON_PATH='./datasets/dataset/acoustic_condition/test/'
SORTED_LIST_PATH='./datasets/sorted_list/'
TEST_WAV_SORTED_LIST=SORTED_LIST_PATH+'test_wav_sorted_list.scp'
TEST_CON_SORTED_LIST=SORTED_LIST_PATH+'test_con_sorted_list.scp'
LOG_FILE='./log.txt'
SAMPLE_RATE=16000
ARGMAX_SAMPLES=2000

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
    parser.add_argument('--wav_out_path', type=str, required=True, help='Which path to save the generated waveforms.')
    parser.add_argument('--restore_from', type=str, required=True, help='Which model checkpoint to generate from')

    args = parser.parse_args()

    return args

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
    else:
        print(" No checkpoint found.")

def main():

    print "Generation started at:", datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M')

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
    WAV_OUT_PATH=args.wav_out_path
    RESTORE_FROM=args.restore_from
    H0_MULT=2 if RNN_TYPE=='LSTM' else 1

    assert SEQ_LEN % CON_FRAME_SIZE == 0,\
        'seq_len should be divisible by con_frame_size'
    assert CON_FRAME_SIZE % BIG_FRAME_SIZE == 0,\
        'con_frame_size should be divisible by big_frame_size'
    assert BIG_FRAME_SIZE % FRAME_SIZE == 0,\
        'big_frame_size should be divisible by frame_size'

    if os.path.exists(TEST_WAV_SORTED_LIST) and os.path.exists(TEST_CON_SORTED_LIST):
        print 'Test sorted list already exists!'
    else:
        if not os.path.exists(SORTED_LIST_PATH):
            os.makedirs(SORTED_LIST_PATH)
        print 'generating test sorted list...'
        generate_sorted_list(TEST_WAV_PATH,TEST_CON_PATH,SAMPLE_RATE,TEST_WAV_SORTED_LIST,TEST_CON_SORTED_LIST)
        print 'Done.'

    if not os.path.exists(WAV_OUT_PATH):
        os.makedirs(WAV_OUT_PATH)

    data_feeder=load_data(
        TEST_WAV_SORTED_LIST,
        TEST_CON_SORTED_LIST,
        BATCH_SIZE,
        SEQ_LEN,
        CON_FRAME_SIZE,
        CON_DIM,
        Q_LEVELS,
        Q_TYPE,
        SAMPLE_RATE)

    con_ph=tf.placeholder(dtype=tf.float32,shape=[None,CON_DIM])
    con_mask_ph=tf.placeholder(dtype=tf.int32,shape=[None,CON_FRAME_SIZE])
    big_wav_ph=tf.placeholder(dtype=tf.int32,shape=[None,BIG_FRAME_SIZE])
    big_mask_ph=tf.placeholder(dtype=tf.int32,shape=[None,BIG_FRAME_SIZE])
    wav_ph=tf.placeholder(dtype=tf.int32,shape=[None,FRAME_SIZE])
    mask_ph=tf.placeholder(dtype=tf.int32,shape=[None,FRAME_SIZE])
    wav_t_ph=tf.placeholder(dtype=tf.int32,shape=[None,])
    mask_t_ph=tf.placeholder(dtype=tf.int32,shape=[None,])

    con_h0_ph=tf.placeholder(dtype=tf.float32,shape=[None,H0_MULT*sum(RNN_HIDDEN_DIM[0])])
    big_h0_ph=tf.placeholder(dtype=tf.float32,shape=[None,H0_MULT*sum(RNN_HIDDEN_DIM[1])])
    h0_ph=tf.placeholder(dtype=tf.float32,shape=[None,H0_MULT*sum(RNN_HIDDEN_DIM[2])])

    con_frame_level_output_ph=tf.placeholder(dtype=tf.float32,shape=[None,1,RNN_HIDDEN_DIM[0][-1]])
    big_frame_level_output_ph=tf.placeholder(dtype=tf.float32,shape=[None,1,RNN_HIDDEN_DIM[1][-1]])
    frame_level_output_ph=tf.placeholder(dtype=tf.float32,shape=[None,RNN_HIDDEN_DIM[2][-1]])
    prev_samples_ph=tf.placeholder(dtype=tf.int32,shape=[None,FRAME_SIZE])
    sample_level_output_ph=tf.placeholder(dtype=tf.float32,shape=[None,Q_LEVELS])

    reset_ph=tf.placeholder(dtype=tf.int32,shape=[])
    argmax_ph=tf.placeholder(dtype=tf.int32,shape=[])

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

        con_frame_level_output,con_h0=net.con_frame_level_rnn(
            con_ph,con_mask_ph,con_h0_ph,reset_ph)
        big_frame_level_output,big_h0=net.big_frame_level_rnn(
            big_wav_ph,big_mask_ph,con_frame_level_output_ph,big_h0_ph,reset_ph)
        frame_level_output,h0=net.frame_level_rnn(
            wav_ph,mask_ph,big_frame_level_output_ph,h0_ph,reset_ph)
        sample_level_output=net.sample_level_predictor(
            frame_level_output_ph,prev_samples_ph)
        new_sample,ce_loss_t,accuracy_t=net.create_generator(
            sample_level_output_ph,wav_t_ph,mask_t_ph,argmax_ph)


    sess=tf.Session(config=tf.ConfigProto(log_device_placement=False))

    saver=tf.train.Saver(var_list=tf.trainable_variables(),max_to_keep=400)
    load_model(saver,sess,RESTORE_FROM)

    try:
        ce_loss_list=[]
        accuracy_list=[]

        wav_mat_list=[]
        mask_mat_list=[]

        samples_number=0
        count=0

        print 'Generation!'
        start_time=time()

        for _wav_batch,_mask_batch,_con_batch,_reset_batch,_end_batch,_end_epoch in data_feeder:

            if _reset_batch==1:
                CON_H0=np.zeros((BATCH_SIZE,H0_MULT*sum(RNN_HIDDEN_DIM[0])))
                BIG_H0=np.zeros((BATCH_SIZE,H0_MULT*sum(RNN_HIDDEN_DIM[1])))
                H0=np.zeros((BATCH_SIZE,H0_MULT*sum(RNN_HIDDEN_DIM[2])))

                samples=np.full((_wav_batch.shape[0],CON_FRAME_SIZE),np.int32((Q_LEVELS-1)//2),dtype='int32')

                cumulative_ce_loss=np.zeros((_wav_batch.shape[0],),dtype='float32')
                cumulative_accuracy=np.zeros((_wav_batch.shape[0],),dtype='float32')
                cumulative_mask=np.zeros((_wav_batch.shape[0],),dtype='int32')


                mask_gen=_mask_batch[:,CON_FRAME_SIZE:]
                index=CON_FRAME_SIZE
            else:
                mask_gen=np.concatenate([mask_gen,_mask_batch[:,CON_FRAME_SIZE:]],axis=1)

            for t in xrange(CON_FRAME_SIZE,SEQ_LEN+CON_FRAME_SIZE):
                if t%CON_FRAME_SIZE==0:
                    CON_FRAME_OUTPUT,CON_H0=sess.run([con_frame_level_output,con_h0],
                        feed_dict={con_ph:_con_batch[:,(t//CON_FRAME_SIZE)*CON_DIM:(t//CON_FRAME_SIZE+1)*CON_DIM],
                                    con_mask_ph:_mask_batch[:,t:t+CON_FRAME_SIZE],
                                    con_h0_ph:CON_H0,
                                    reset_ph:_reset_batch})

                if t%BIG_FRAME_SIZE==0:
                    BIG_FRAME_OUTPUT,BIG_H0=sess.run([big_frame_level_output,big_h0],
                        feed_dict={big_wav_ph:samples[:,index-BIG_FRAME_SIZE:index],
                                    big_mask_ph:_mask_batch[:,t:t+BIG_FRAME_SIZE],
                                    con_frame_level_output_ph:CON_FRAME_OUTPUT[:,(t/BIG_FRAME_SIZE)%(CON_FRAME_SIZE/BIG_FRAME_SIZE)].reshape(_wav_batch.shape[0],1,-1),
                                    big_h0_ph:BIG_H0,
                                    reset_ph:_reset_batch})

                if t%FRAME_SIZE==0:
                    FRAME_OUTPUT,H0=sess.run([frame_level_output,h0],
                        feed_dict={wav_ph:samples[:,index-FRAME_SIZE:index],
                                    mask_ph:_mask_batch[:,t:t+FRAME_SIZE],
                                    big_frame_level_output_ph:BIG_FRAME_OUTPUT[:,(t/FRAME_SIZE)%(BIG_FRAME_SIZE/FRAME_SIZE)].reshape(_wav_batch.shape[0],1,-1),
                                    h0_ph:H0,
                                    reset_ph:_reset_batch})

                SAMPLE_OUTPUT=sess.run(sample_level_output,
                    feed_dict={frame_level_output_ph:FRAME_OUTPUT[:,t%FRAME_SIZE],
                                prev_samples_ph:samples[:,index-FRAME_SIZE:index]})

                if index<ARGMAX_SAMPLES+CON_FRAME_SIZE:
                    NEW_SAMPLE,CE_LOSS,ACCURACY=sess.run([new_sample,ce_loss_t,accuracy_t],
                        feed_dict={sample_level_output_ph:SAMPLE_OUTPUT,
                                    wav_t_ph:_wav_batch[:,t],
                                    mask_t_ph:_mask_batch[:,t],
                                    argmax_ph:1})
                else:
                    NEW_SAMPLE,CE_LOSS,ACCURACY=sess.run([new_sample,ce_loss_t,accuracy_t],
                        feed_dict={sample_level_output_ph:SAMPLE_OUTPUT,
                                    wav_t_ph:_wav_batch[:,t],
                                    mask_t_ph:_mask_batch[:,t],
                                    argmax_ph:0})

                cumulative_ce_loss+=CE_LOSS
                cumulative_accuracy+=ACCURACY
                cumulative_mask+=_mask_batch[:,t]

                samples=np.concatenate([samples,NEW_SAMPLE],axis=1)

                index+=1

            if _end_batch==1:
                ce_loss_list.extend(list(cumulative_ce_loss/cumulative_mask))
                accuracy_list.extend(list(cumulative_accuracy/cumulative_mask))

                wav_mat_list.append(samples[:,CON_FRAME_SIZE:])
                mask_mat_list.append(mask_gen)

        ce_loss=np.mean(ce_loss_list)
        accuracy=np.mean(accuracy_list)

        fid=open(TEST_WAV_SORTED_LIST,'r')
        gen_id_list=fid.readlines()
        fid.close()

        for i in xrange(len(wav_mat_list)):
            samples_number+=wav_mat_list[i].shape[0]*wav_mat_list[i].shape[1]
            for j in xrange(wav_mat_list[i].shape[0]):
                samplei=wav_mat_list[i][j]
                maski=mask_mat_list[i][j]
                samplei=samplei[0:len(np.where(maski==1)[0])]
                if Q_TYPE=='mu-law':
                    from datasets.audio_reader import mu2linear
                    samplei=mu2linear(samplei,Q_LEVELS)

                sf.write(WAV_OUT_PATH+os.sep+gen_id_list[count].split()[0].split('/')[-1],samplei,SAMPLE_RATE,'PCM_16')
                count+=1

        generation_time=time()-start_time

        log="{} samples generated in {} hours.\nThe time of generating 1 second speech is {} seconds.\n"
        log+="Performance:\n\tloss:{:.4f}\taccuracy:{:.4f}%\n"
        log=log.format(len(gen_id_list),generation_time/3600,generation_time/samples_number*SAMPLE_RATE,ce_loss,accuracy*100)
        print log

        fid_log=open(LOG_FILE,'a')
        fid_log.write(log)
        fid_log.close()

        print "Generation ended at:", datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M')

    except KeyboardInterrupt:
        # Introduce a line break after ^C is displayed so save message
        # is on its own line.
        print()

if __name__=='__main__':
    main()