#!/usr/bin/env python
# Copyright 2017 Uber Technologies, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import os
import signal
import errno
import tensorflow as tf
import numpy as np

from tensorflow import keras

import fix_paths
import elf
import elf.tensorflow

layers = tf.layers

tf.logging.set_verbosity(tf.logging.INFO)


def conv_model(feature, target, mode):
    """2-layer convolution model."""
    # Convert the target to a one-hot tensor of shape (batch_size, 10) and
    # with a on-value of 1 for each one-hot vector of length 10.
    target = tf.one_hot(tf.cast(target, tf.int32), 10, 1, 0)

    # Reshape feature to 4d tensor with 2nd and 3rd dimensions being
    # image width and height final dimension being the number of color channels.
    feature = tf.reshape(feature, [-1, 28, 28, 1])

    # First conv layer will compute 32 features for each 5x5 patch
    with tf.variable_scope('conv_layer1'):
        h_conv1 = layers.conv2d(feature, 32, kernel_size=[5, 5],
                                activation=tf.nn.relu, padding="SAME")
        h_pool1 = tf.nn.max_pool(
            h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # Second conv layer will compute 64 features for each 5x5 patch.
    with tf.variable_scope('conv_layer2'):
        h_conv2 = layers.conv2d(h_pool1, 64, kernel_size=[5, 5],
                                activation=tf.nn.relu, padding="SAME")
        h_pool2 = tf.nn.max_pool(
            h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        # reshape tensor into a batch of vectors
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])

    # Densely connected layer with 1024 neurons.
    h_fc1 = layers.dropout(
        layers.dense(h_pool2_flat, 1024, activation=tf.nn.relu),
        rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Compute logits (1 per class) and compute loss.
    logits = layers.dense(h_fc1, 10, activation=None)
    loss = tf.losses.softmax_cross_entropy(target, logits)

    return tf.argmax(logits, 1), loss


def main(controller_address):
    worker = elf.Worker(controller_address)

    # Keras automatically creates a cache directory in ~/.keras/datasets for
    # storing the downloaded MNIST data. This creates a race
    # condition among the workers that share the same filesystem. If the
    # directory already exists by the time this worker gets around to creating
    # it, ignore the resulting exception and continue.
    cache_dir = os.path.join(os.path.expanduser('~'), '.keras', 'datasets')
    if not os.path.exists(cache_dir):
        try:
            os.mkdir(cache_dir)
        except OSError as e:
            if e.errno == errno.EEXIST and os.path.isdir(cache_dir):
                pass
            else:
                raise

    # Download and load MNIST dataset.
    (x_train, y_train), (x_test, y_test) = \
        keras.datasets.mnist.load_data('MNIST-data')

    # The shape of downloaded data is (-1, 28, 28), hence we need to reshape it
    # into (-1, 784) to feed into our network. Also, need to normalize the
    # features between 0 and 1.
    x_train = np.reshape(x_train, (-1, 784)) / 255.0
    x_test = np.reshape(x_test, (-1, 784)) / 255.0

    # Build model...
    with tf.name_scope('input'):
        image = tf.placeholder(tf.float32, [None, 784], name='image')
        label = tf.placeholder(tf.float32, [None], name='label')
    predict, loss = conv_model(image, label, tf.estimator.ModeKeys.TRAIN)

    opt = tf.train.RMSPropOptimizer(0.001)

    opt = elf.tensorflow.ElasticOptimizer(worker, opt)

    global_step = tf.train.get_or_create_global_step()
    train_op = opt.minimize(loss, global_step=global_step)
    opt.commit_and_join()

    hooks = [
        tf.train.LoggingTensorHook(tensors={'step': global_step, 'loss': loss},
                                   every_n_iter=50),
    ]

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    # demo
    def handler(signal, frame):
        print('Received Interrupt, request leaving...')
        worker.leave()
    signal.signal(signal.SIGINT, handler)

    # The MonitoredTrainingSession takes care of session initialization,
    # restoring from a checkpoint, saving to a checkpoint, and closing when done
    # or an error occurs.
    with tf.train.MonitoredTrainingSession(hooks=hooks,
                                           config=config) as mon_sess:
        opt._session = mon_sess
        for indices in worker.shard_generator(range(len(x_train)), batch_size=100):
            image_ = x_train[indices]
            label_ = y_train[indices]
            mon_sess.run(train_op, feed_dict={image: image_, label: label_})

if __name__ == "__main__":
    import sys
    main(sys.argv[1])
