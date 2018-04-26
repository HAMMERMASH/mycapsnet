import tensorflow as tf
import numpy as np
import scipy.misc as misc
from layer import ip_layer, conv_layer, pool_layer
from read_data import read_mnist
import time
import os
#os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_IS'
#os.environ['CUDA_VISIBLE_DEVICE'] = '0'

MNIST_DIR = './'
NUM_TRAIN = 60000
NUM_TEST = 10000
BATCH_SIZE = 2048
IMG_SIZE = 28
INPUT_SIZE = 28
INIT_LR = 0.001
LR_DECAY = 0.95
NUM_EPOCH = 100
MOMENTUM = 0.9
RECON_LAMBDA = 0.0005

LAMBDA = 0.5
M_MAX = 0.9
M_MIN = 0.1

def squash(x):
    norm = tf.reduce_sum(tf.square(x), axis=2, keep_dims=True)
    norm_factor = tf.stop_gradient(norm/(1+norm)/tf.sqrt(norm+0.001))
    x = norm_factor*x

    return x
"""
def capsnet(x):

    with tf.variable_scope('capsule', reuse=tf.AUTO_REUSE):
        conv1 = conv_layer('conv1',x,[9,9,1,256],padding='VALID',activation='relu') 
        conv2 = conv_layer('conv2',conv1,[9,9,256,256],padding='VALID',stride=[1,2,2,1],activation=None)

        primary_blob = tf.reshape(conv2,[BATCH_SIZE,1152,8])
        primary_blob = squash(primary_blob)

        primary_caps = tf.split(primary_blob, num_or_size_splits=1152, axis=1)
        primary_output = []
        second_caps = []
        for i in range(1152):
            output_i = tf.matmul(tf.reshape(primary_caps[i], [BATCH_SIZE,8]),tf.get_variable('W_{}'.format(i),[8,160],tf.float32))
            output_i = tf.reshape(output_i,[BATCH_SIZE,10,16])
            primary_output.append(output_i)
            output_i = tf.transpose(output_i,[0,2,1])
            output_i = tf.reshape(output_i,[-1,10])
            c_i = tf.nn.softmax(tf.get_variable('b_{}'.format(i),[1,10],tf.float32,trainable=False, initializer=tf.constant_initializer(0)))
            output_i = output_i * c_i
            output_i = tf.reshape(output_i, [BATCH_SIZE,16,10])
            output_i = tf.transpose(output_i,[0,2,1])
            second_caps.append(output_i)
        
        second_caps = tf.reduce_sum(tf.stack(second_caps), axis=0)
        second_caps_norm = squash(second_caps)

        update_ops = []
        for i in range(1152):
            deltas = tf.reduce_sum(primary_output[i]*second_caps_norm,axis=2)
            deltas = tf.reduce_mean(deltas, axis=0, keep_dims=True)
            b_i = tf.get_variable('b_{}'.format(i),[1,10],tf.float32,trainable=False)
            update_ops.append(tf.assign(b_i, b_i+deltas))

        return second_caps_norm, update_ops

"""
def capsnet(x, route_iter):

    with tf.variable_scope('capsule', reuse=tf.AUTO_REUSE):
        conv1 = conv_layer('conv1',x,[9,9,1,256],padding='VALID',activation='relu') 
        conv2 = conv_layer('conv2',conv1,[9,9,256,256],padding='VALID',stride=[1,2,2,1],activation=None)

        primary_blob = tf.reshape(conv2,[BATCH_SIZE,1152,8])
        primary_blob = squash(primary_blob)

        Bs = tf.zeros([1152,10], tf.float32)
        for i in range(route_iter):
            
            Ws = tf.get_variable('Ws', [1152,8,160], tf.float32)
            primary_blob_transpose = tf.transpose(primary_blob, [1,0,2])
            second_logits = tf.matmul(primary_blob_transpose, Ws)
            second_logits = tf.transpose(second_logits, [1,0,2])
            second_logits = tf.reshape(second_logits, [BATCH_SIZE,1152,10,16])
            Cs = tf.nn.softmax(Bs)
            Cs = tf.reshape(Cs, [1,1152,10,1])
            second_logits = second_logits*Cs
            second_input = tf.reduce_sum(second_logits, axis=1)
            second_output = squash(second_input)
            Bs = Bs+tf.reduce_sum(second_logits*tf.reshape(second_output, [BATCH_SIZE,1,10,16]), axis=(0,3))/BATCH_SIZE
            
        return second_output 


def margin_loss(caps_norm, labels):
    
    T_k = tf.one_hot(labels, 10)
    lambda_T_k = (1-T_k)*LAMBDA

    loss = tf.reduce_sum(tf.square(tf.maximum(M_MAX-caps_norm, 0))*T_k + tf.square(tf.maximum(caps_norm-M_MIN, 0))*lambda_T_k, axis=1)

    return loss


def train():
    
    with tf.Graph().as_default():
      
        img = tf.placeholder(tf.float32,(BATCH_SIZE,INPUT_SIZE,INPUT_SIZE,1))
        label = tf.placeholder(tf.int32,(BATCH_SIZE,))

        caps = capsnet(img, 3)
        caps_norm = tf.sqrt(tf.reduce_sum(tf.square(caps), axis=2))
        loss = tf.reduce_mean(margin_loss(caps_norm, label))
        
        global_step = tf.Variable(0, trainable=False)   
        decay_step = int(NUM_TRAIN/BATCH_SIZE)
        learning_rate = tf.train.exponential_decay(INIT_LR, global_step,
                                                   decay_step, LR_DECAY, staircase=True)
        optimizer = tf.train.AdamOptimizer(INIT_LR)
        train_op = optimizer.minimize(loss,global_step=global_step)

        initializer = tf.global_variables_initializer()
        
        with tf.Session() as sess:
            
            sess.run(initializer)
            print('reading data...')
            train_data,train_label,test_data,test_label = read_mnist(MNIST_DIR)
            if INPUT_SIZE != IMG_SIZE:
                resize_train = []
                resize_test = []
                for i in range(NUM_TRAIN):
                    resize_img = misc.imresize(train_data[i],(INPUT_SIZE,INPUT_SIZE),'bilinear')
                    resize_train.append(resize_img)
                for i in range(NUM_TEST):
                    resize_img = misc.imresize(test_data[i],(INPUT_SIZE,INPUT_SIZE),'bilinear')
                    resize_test.append(resize_img)
                train_data = np.stack(resize_train)
                test_data = np.stack(resize_test)
            train_data = np.expand_dims(train_data,axis=-1)
            test_data = np.expand_dims(test_data,axis=-1)

            print('training...')
            # train
            num_iter = decay_step*NUM_EPOCH
            epoch_step = decay_step
            batch_inds = np.arange(0,NUM_TRAIN,1)
            np.random.shuffle(batch_inds)
            cur_ind = 0
            epoch = 0
            loss_avg = 0
            epoch_iter = 0
            for i in range(num_iter):
            
                # prepare train batch
                start = time.time()
                cur_to = cur_ind+BATCH_SIZE
                cur_data = None
                cur_label = None
                if cur_to >= NUM_TRAIN:
                    cur_data = train_data[batch_inds[cur_ind:]]
                    cur_label = train_label[batch_inds[cur_ind:]]
                    pad_to = cur_to-NUM_TRAIN
                    cur_data = np.concatenate((cur_data,train_data[batch_inds[0:pad_to]]))
                    cur_label = np.concatenate((cur_label,train_label[batch_inds[0:pad_to]]))
                
                else:
                    cur_data = train_data[batch_inds[cur_ind:cur_to]]
                    cur_label = train_label[batch_inds[cur_ind:cur_to]]

                cur_ind = cur_to
                
                # train batch
                _, loss_display = sess.run([train_op, loss], feed_dict={img:cur_data,label:cur_label})
                loss_avg += loss_display
                epoch_iter += 1
                end = time.time()
                if cur_ind >= NUM_TRAIN:
                    np.random.shuffle(batch_inds)
                    cur_ind = 0

                    epoch += 1
 
                    print("epoch: {}, loss: {}".format(epoch,loss_avg/epoch_iter))
                    print('testing...')
                    test_iter = int(NUM_TEST/BATCH_SIZE)
                    predicts = []
                    for i in range(test_iter):
                        test_batch = test_data[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
                        predict_batch = sess.run(caps_norm,feed_dict={img:test_batch})
                        predicts.append(predict_batch)

                    if test_iter*BATCH_SIZE < NUM_TEST:
                        num_tested = test_iter*BATCH_SIZE
                        test_batch = test_data[num_tested:]
                        pad = np.zeros((BATCH_SIZE-NUM_TEST+num_tested,INPUT_SIZE,INPUT_SIZE,1))
                        test_batch = np.concatenate((test_batch,pad))
                        predict_batch = sess.run(caps_norm,feed_dict={img:test_batch})
                        predicts.append(predict_batch[:NUM_TEST-num_tested])
                    
                    predicts = np.concatenate(predicts)
                    predicts = np.argmax(predicts,axis=1)
                    accuracy = np.sum(np.equal(predicts,test_label))/predicts.shape[0]
                    print('accuracy: {}'.format(accuracy))
                    loss_avg = 0
                    epoch_iter = 0
            
            # test
            print('testing...')
            test_iter = int(NUM_TEST/BATCH_SIZE)
            predicts = []
            for i in range(test_iter):
                test_batch = test_data[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
                predict_batch = sess.run(caps_norm,feed_dict={img:test_batch})
                predicts.append(predict_batch)

            if test_iter*BATCH_SIZE < NUM_TEST:
                num_tested = test_iter*BATCH_SIZE
                test_batch = test_data[num_tested:]
                pad = np.zeros((BATCH_SIZE-NUM_TEST+num_tested,INPUT_SIZE,INPUT_SIZE,1))
                test_batch = np.concatenate((test_batch,pad))
                predict_batch = sess.run(logits,feed_dict={img:test_batch})
                predicts.append(predict_batch[:NUM_TEST-num_tested])
            
            predicts = np.concatenate(predicts)
            predicts = np.argmax(predicts,axis=1)
            accuracy = np.sum(np.equal(predicts,test_label))/predicts.shape[0]
            print('accuracy: {}'.format(accuracy))

                   

if __name__ == '__main__':
    
    train()



                


