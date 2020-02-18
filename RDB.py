import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import os
import time
import cv2


def res_denseblock(self, input_tensor, scope_name):
    with tf.variable_scope(scope_name):
        RDBs = []
        x_input = input_tensor

        """
        n_rdb = 5 ( RDB number )
        n_rdb_conv = 6 ( per RDB conv layer )
        """

        for k in range(3):
            if k == 0:
                with tf.variable_scope('RDB_' + str(k)):
                    layers = []
                    layers.append(input_tensor)

                    # self.x = slim.conv2d(input_tensor, 32, 3, 1, scope='block_{:d}_conv'.format(k))
                    x = self.leakyRelu(slim.conv2d(input_tensor, 32, 3, 1, scope='block_{:d}_conv'.format(k)))

                    layers.append(x)

                    for i in range(1, 2):
                        x = tf.concat(layers, axis=-1)

                        # self.x = slim.conv2d(x, 32, 3, 1, scope = 'block_{:d}_conv_2'.format(i))
                        x = self.leakyRelu(slim.conv2d(x, 32, 3, 1, scope='block_{:d}_conv_2'.format(i)))

                        layers.append(x)
            else:
                with tf.variable_scope('RDB_' + str(k)):
                    layers = []
                    layers.append(input_tensor)

                    x = self.leakyRelu(slim.conv2d(input_tensor, 32, 3, 1, scope='block_{:d}_conv_3'.format(k)))

                    layers.append(x)

                    for i in range(1, 2):
                        x = tf.concat(layers, axis=-1)

                        x = self.leakyRelu(slim.conv2d(x, 32, 3, 1, scope='block_{:d}_conv_4'.format(i)))

                        layers.append(x)

                    # Local feature fusion
                    x = tf.concat(layers, axis=-1)
                    x = slim.conv2d(x, 32, 3, 1, scope='conv_last')

                    # Local residual learning
                    # x = input_tensor + x

                    RDBs.append(x)
                    input_tensor = x
        with tf.variable_scope('GFF_1x1'):
            x = tf.concat(RDBs, axis=-1)
            x = slim.conv2d(x, 32, kernel_size=1, stride=1, scope='conv')

        with tf.variable_scope('GFF_3x3'):
            x = slim.conv2d(x, 32, kernel_size=3, stride=1, scope='conv')

        # Global residual learning
        output = input_tensor + x

        return output