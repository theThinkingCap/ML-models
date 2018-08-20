import tensorflow as tf
import numpy as np
import math

def fc_layer(inputs, output_units, kernel=None, bias=None, activation=tf.nn.leaky_relu):
    n_feat = inputs.get_shape()[1]
    if kernel is None:
        kernel = tf.Variable(tf.truncated_normal([n_feat, output_units], stddev=1.0 / math.sqrt(float(n_feat))), name="Weights")
    if bias is None:
        bias = tf.Variable(tf.zeros([n_feat, output_units]), name="Biases")

    cell_body = tf.add(tf.matmul(inputs,kernel), bias)
    activated = activation(cell_body)

    return activated