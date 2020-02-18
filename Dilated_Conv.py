import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import os
import time
import cv2


def MAEB1(self, input_x, scope_name, dilated_factors=3):
    '''MAEB: multi-scale aggregation and enhancement block
        Params:
            input_x: input data
            scope_name: the scope name of the MAEB (customer definition)
            dilated_factor: the maximum number of dilated factors(default=3, range from 1 to 3)

        Return:
            return the output the MAEB

        Input shape:
            4D tensor with shape '(batch_size, height, width, channels)'

        Output shape:
            4D tensor with shape '(batch_size, height, width, channels)'
    '''
    dilate_c = []
    with tf.variable_scope(scope_name):
        for i in range(1, dilated_factors + 1):
            d1 = self.leakyRelu(slim.conv2d(input_x, self.channel_dim, 3, 1, rate=i, activation_fn=None, scope='d1'))
            d2 = self.leakyRelu(slim.conv2d(d1, self.channel_dim, 3, 1, rate=i, activation_fn=None, scope='d2'))
            dilate_c.append(d2)

        add = tf.add_n(dilate_c)
        shape = add.get_shape().as_list()
        output = self.SEBlock(add, shape[-1], reduce_dim=int(shape[-1] / 4))
        return output