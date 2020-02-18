import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import os
import time
import cv2


def _residual_block(self, input_tensor, scope_name):
    output = None
    with tf.variable_scope(scope_name):
        for i in range(6):
            if i == 0:
                # self.conv_1 = slim.conv2d(input_tensor, 32, 3, 1, scope='block_{:d}_conv_1'.format(i))
                relu_1 = self.leakyRelu(slim.conv2d(input_tensor, 32, 3, 1, scope='block_{:d}_conv_1'.format(i)))
                output = relu_1
                input_tensor = output
            else:
                # self.conv_1 = slim.conv2d(input_tensor, 32, 3, 1, scope='block_{:d}_conv_1'.format(i))
                relu_1 = self.leakyRelu(slim.conv2d(input_tensor, 32, 3, 1, scope='block_{:d}_conv_1'.format(i)))
                # self.conv_2 = slim.conv2d(input_tensor, 32, 3, 1, scope='block_{:d}_conv_2'.format(i))
                relu_2 = self.leakyRelu(slim.conv2d(input_tensor, 32, 3, 1, scope='block_{:d}_conv_2'.format(i)))

                output = self.leakyRelu(tf.add(relu_2, input_tensor))
                input_tensor = output

    return output