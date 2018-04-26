import tensorflow as tf
import numpy as np

def create_variable(name,shape,initializer):

    return tf.get_variable(name = name,shape = shape,
        initializer = initializer,dtype = tf.float32)

def batch_normalize(name,tensor,phase_train):

    with tf.variable_scope(name):

        return tf.layers.batch_normalization(inputs = tensor,
            training = phase_train)

def conv_layer(
    name,
    bottom,
    shape,
    padding = 'SAME',
    stride = [1,1,1,1],
    activation = 'relu',
    batch_normalization = False,
    phase_train = True):

    with tf.variable_scope(name):

        kernal = create_variable(name = 'weights',shape = shape,
            initializer = tf.contrib.layers.xavier_initializer_conv2d(uniform = False))
        conv = tf.nn.conv2d(bottom,kernal,stride,padding = padding)
        biases = create_variable(name = 'biases',shape = [shape[-1]],
            initializer = tf.contrib.layers.xavier_initializer(uniform = False))
        preactivation = tf.nn.bias_add(conv,biases)
        if batch_normalization:
            preactivation = batch_normalize(name = 'bn',
                tensor = preactivation,
                phase_train = phase_train)
        if activation == 'relu':
            relu = tf.nn.relu(preactivation,name = name)
            return relu
        if activation == 'sigmoid':
            sigmoid = tf.nn.sigmoid(preactivation, name=name)

        return preactivation

def atrous_conv(
    name,bottom,shape,
    padding = 'SAME',
    stride = [1,1,1,1],
    activation = 'Relu',
    rate = 6,
    batch_normalization = False,
    phase_train = True):

    with tf.variable_scope(name):
        kernal = create_variable(name = 'weights',shape = shape,
            initializer = tf.contrib.layers.xavier_initializer_conv2d(uniform = False))
        conv = tf.nn.atrous_conv2d(bottom,kernal,rate = rate,padding = padding)
        biases = create_variable(name = 'biases',shape = [shape[-1]],
            initializer = tf.contrib.layers.xavier_initializer(uniform = False))
        preactivation = tf.nn.bias_add(conv,biases)
        if batch_normalization:
            preactivation = batch_normalize(name = 'bn',
                tensor = preactivation,
                phase_train = phase_train)
        if activation == 'Relu':
            relu = tf.nn.relu(preactivation,name = name) 
            return relu
 
        return preactivation


def ip_layer(name,bottom,top,activation=None,no_bias=False):
    with tf.variable_scope(name):
        shape = bottom.get_shape().as_list()
        dim = np.prod(shape[1:])
        reshape = tf.reshape(bottom,[-1,dim])
        weights = create_variable(name = 'weights',shape = [dim,top],
            initializer = tf.contrib.layers.xavier_initializer(uniform = False))
        biases = create_variable(name = 'biases',shape = [top],
            initializer = tf.contrib.layers.xavier_initializer())

        logits = tf.nn.bias_add(tf.matmul(reshape,weights),biases)
        if activation == 'relu':
            logits = tf.nn.relu(logits,name=name)
        if activation == 'sigmoid':
            logits = tf.nn.sigmoid(logits,name=name)


        return logits

def pool_layer(name,bottom,ksize = [1,2,2,1],strides = [1,2,2,1]):
    return tf.nn.avg_pool(bottom,ksize = ksize,
        strides = strides,padding = 'SAME',name = name)

def l2_normalization(name,inputs,scaling = False):

    with tf.variable_scope(name):
        inputs_shape = inputs.get_shape()
        inputs_rank = inputs_shape.ndims

        norm_dim = tf.range(inputs_rank - 1,inputs_rank)
        params_shape = inputs_shape[-1:]

        outputs = tf.nn.l2_normalize(inputs,norm_dim)

        if scaling:
            scale = tf.Variable(tf.ones(params_shape) * 20,trainable = True)
            outputs = tf.multiply(outputs,scale)

        return outputs
