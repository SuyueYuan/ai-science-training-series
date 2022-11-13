# Copyright 2019 Uber Technologies, Inc. All Rights Reserved.
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

import tensorflow as tf
import argparse
import numpy as np
import math, os, sys
import time

# This limits the amount of memory used:
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=2"

parser = argparse.ArgumentParser(description='TensorFlow MNIST Horovod')
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='input batch size for training (default: 256)')
parser.add_argument('--epochs', type=int, default=16, metavar='N',
                    help='number of epochs to train (default: 16)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--device', default='gpu',
                    help='Wheter this is running on cpu or gpu')

args = parser.parse_args()

# This control parallelism in Tensorflow
parallel_threads = 1
tf.config.threading.set_inter_op_parallelism_threads(parallel_threads)
tf.config.threading.set_intra_op_parallelism_threads(parallel_threads)

#---------------------------------------------------
# Initialize Horovd and Parallel Threads
#---------------------------------------------------
hvd.init()
print("# I am rank %d of %d" %(hvd.rank(), hvd.size()))
parallel_threads = parallel_threads//hvd.size()
os.environ['OMP_NUM_THREADS'] = str(parallel_threads)
num_parallel_readers = parallel_threads

# Assign GPUs to each rank, but I have no access to multiple GPUs
# gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

#---------------------------------------------------
# Load dataset
#---------------------------------------------------

(mnist_images, mnist_labels), (x_test, y_test) = \
    tf.keras.datasets.mnist.load_data(path='mnist.npz')

dataset = tf.data.Dataset.from_tensor_slices(
    (tf.cast(mnist_images[..., tf.newaxis] / 255.0, tf.float32),
             tf.cast(mnist_labels, tf.int64))
)
test_dset = tf.data.Dataset.from_tensor_slices(
    (tf.cast(x_test[..., tf.newaxis] / 255.0, tf.float32),
             tf.cast(y_test, tf.int64))
)

dataset = dataset.repeat().shuffle(10000).batch(args.batch_size)
dataset = dataset.shard(num_shards=hvd.size(), index=hvd.rank())
test_dset  = test_dset.repeat().batch(args.batch_size)

options = tf.data.Options()
options.threading.private_threadpool_size = parallel_threads
dataset = dataset.with_options(options)
test_dset = test_dset.with_options(options)

nsamples = len(list(dataset))
ntests = len(list(test_dset))
nstep = nsamples//args.batch_size
ntest_step = ntests//args.batch_size

metrics={}
metrics['train_acc'] = []
metrics['valid_acc'] = []
metrics['train_loss'] = []
metrics['valid_loss'] = []
metrics['time_per_epochs'] = []

#----------------------------------------------------
# Model
#----------------------------------------------------
mnist_model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, [3, 3], activation='relu'),
    tf.keras.layers.Conv2D(64, [3, 3], activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation='softmax')
])
loss = tf.losses.SparseCategoricalCrossentropy()

# Scale the learning rate
opt = tf.optimizers.Adam(args.lr*hvd.size())

checkpoint_dir = './checkpoints/tf2_mnist'
checkpoint = tf.train.Checkpoint(model=mnist_model, optimizer=opt)

#------------------------------------------------------------------
# Training
#------------------------------------------------------------------
@tf.function
def training_step(images, labels):
    with tf.GradientTape() as tape:
        probs = mnist_model(images, training=True)
        loss_value = loss(labels, probs)
        pred = tf.math.argmax(probs, axis=1)
        equality = tf.math.equal(pred, labels)
        accuracy = tf.math.reduce_mean(tf.cast(equality, tf.float32))
    
    # Wrap the gradient tape
    tape = hvd.DistributedGradientTape(tape)
    
    grads = tape.gradient(loss_value, mnist_model.trainable_variables)
    opt.apply_gradients(zip(grads, mnist_model.trainable_variables))
    return loss_value, accuracy

    
@tf.function
def validation_step(images, labels):
    probs = mnist_model(images, training=False)
    pred = tf.math.argmax(probs, axis=1)
    equality = tf.math.equal(pred, labels)
    accuracy = tf.math.reduce_mean(tf.cast(equality, tf.float32))
    loss_value = loss(labels, probs)
    # Average the metrics
    total_loss = hvd.allreduce(loss_value, average=True)
    total_acc = hvd.allreduce(accuracy, average=True)
    return total_loss, total_acc


t0 = time.time()

for ep in range(args.epochs):
    hvd.broadcast_variables(network.variables, root_rank=0)
    hvd.broadcast_variables(optimizer.variables(), root_rank=0)
    
    tt0 = time.time()
    
    training_loss = 0.0
    training_acc = 0.0
    for batch, (images, labels) in enumerate(dataset.take(nstep)):
        loss_value, acc = training_step(images, labels)
        training_loss += loss_value/nstep
        training_acc += acc/nstep
    training_loss = hvd.allreduce(training_loss, average=True)
    training_acc = hvd.allreduce(training_acc, average=True)
    
    test_acc = 0.0
    test_loss = 0.0
    for batch, (images, labels) in enumerate(test_dset.take(ntest_step)):
        loss_value, acc = validation_step(images, labels)
        test_acc += acc/ntest_step
        test_loss += loss_value/ntest_step
    test_acc = hvd.allreduce(test_acc, average=True)
    test_loss = hvd.allreduce(test_loss, average=True)
    
    tt1 = time.time()
    
    if (hvd.rank()==0):
        print('E[%d], train Loss: %.6f, training Acc: %.3f, val loss: %.3f, val Acc: %.3f\t Time: %.3f seconds' % (ep, training_loss, training_acc, test_loss, test_acc, tt1 - tt0))
        metrics['train_acc'].append(training_acc.numpy())
        metrics['train_loss'].append(training_loss.numpy())
        metrics['valid_acc'].append(test_acc.numpy())
        metrics['valid_loss'].append(test_loss.numpy())
        metrics['time_per_epochs'].append(tt1 - tt0)
        checkpoint.save(checkpoint_dir)
        np.savetxt("metrics.dat", np.array([metrics['train_acc'], metrics['train_loss'], metrics['valid_acc'], metrics['valid_loss'], metrics['time_per_epochs']]).transpose())

t1 = time.time()

print("Total training time: %s seconds" %(t1 - t0))
