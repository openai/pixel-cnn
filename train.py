"""
Trains a Pixel-CNN++ generative model on CIFAR-10 or Tiny ImageNet data.
Uses multiple GPUs, indicated by the flag --nr-gpu

Example usage:
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_double_cnn.py --nr_gpu 4
"""

import os
import sys
import time
import json
import argparse

import numpy as np
import tensorflow as tf

import pixel_cnn_pp.nn as nn
import pixel_cnn_pp.plotting as plotting
import pixel_cnn_pp.scopes as scopes
from pixel_cnn_pp.model import model_spec
import data.cifar10_data as cifar10_data

# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser()
# data I/O
parser.add_argument('-i', '--data_dir', type=str, default='/tmp/pxpp/data', help='Location for the dataset')
parser.add_argument('-o', '--save_dir', type=str, default='/tmp/pxpp/save', help='Location for parameter checkpoints and samples')
parser.add_argument('-d', '--data_set', type=str, default='cifar', help='Can be either cifar|imagenet')
parser.add_argument('-t', '--save_interval', type=int, default=20, help='Every how many epochs to write checkpoint/samples?')
parser.add_argument('-r', '--load_params', type=int, default=0, help='Restore training from previous model checkpoint? 1 = Yes, 0 = No')
# model
parser.add_argument('-q', '--nr_resnet', type=int, default=5, help='Number of residual blocks per stage of the model')
parser.add_argument('-n', '--nr_filters', type=int, default=256, help='Number of filters to use across the model. Higher = larger model.')
parser.add_argument('-m', '--nr_logistic_mix', type=int, default=10, help='Number of logistic components in the mixture. Higher = more flexible model')
# optimization
parser.add_argument('-l', '--learning_rate', type=float, default=0.001, help='Base learning rate')
parser.add_argument('-e', '--lr_decay', type=float, default=0.999995, help='Learning rate decay, applied every step of the optimization')
parser.add_argument('-b', '--batch_size', type=int, default=12, help='Batch size during training per GPU')
parser.add_argument('-a', '--init_batch_size', type=int, default=100, help='How much data to use for data-dependent initialization.')
parser.add_argument('-p', '--dropout_p', type=float, default=0.5, help='Dropout strength (i.e. 1 - keep_prob). 0 = No dropout, higher = more dropout.')
parser.add_argument('-g', '--nr_gpu', type=int, default=8, help='How many GPUs to distribute the training across?')
# evaluation
parser.add_argument('--sample_batch_size', type=int, default=4, help='How many images to process in paralell during sampling?')
parser.add_argument('--polyak_decay', type=float, default=0.9995, help='Exponential decay rate of the sum of previous model iterates during Polyak averaging')
# reproducibility
parser.add_argument('-s', '--seed', type=int, default=1, help='Random seed to use')
args = parser.parse_args()
print('input args:\n', json.dumps(vars(args), indent=4, separators=(',',':'))) # pretty print args

# -----------------------------------------------------------------------------
# fix random seed for reproducibility
rng = np.random.RandomState(args.seed)
tf.set_random_seed(args.seed)

# create the model
model_opt = { 'nr_resnet': args.nr_resnet, 'nr_filters': args.nr_filters, 'nr_logistic_mix': args.nr_logistic_mix }
model = tf.make_template('model', model_spec)

# data
x_init = tf.placeholder(tf.float32, shape=(args.init_batch_size, 32, 32, 3))

# run once for data dependent initialization of parameters
gen_par = model(x_init, init=True, dropout_p=args.dropout_p, **model_opt)

# get list of all params
all_params = tf.trainable_variables()

# keep track of moving average
ema = tf.train.ExponentialMovingAverage(decay=args.polyak_decay)
maintain_averages_op = tf.group(ema.apply(all_params))

# sample from the model
x_sample = tf.placeholder(tf.float32, shape=(args.sample_batch_size, 32, 32, 3))
gen_par = model(x_sample, ema=ema, dropout_p=0, **model_opt)
new_x_gen = nn.sample_from_discretized_mix_logistic(gen_par, args.nr_logistic_mix)
def sample_from_model(sess):
    x_gen = np.zeros((args.sample_batch_size,32,32,3), dtype=np.float32)
    for yi in range(32):
        for xi in range(32):
            new_x_gen_np = sess.run(new_x_gen, {x_sample: x_gen})
            x_gen[:,yi,xi,:] = new_x_gen_np[:,yi,xi,:].copy()
    return x_gen

# get loss gradients over multiple GPUs
xs = []
grads = []
loss_gen = []
loss_gen_test = []
for i in range(args.nr_gpu):
    xs.append(tf.placeholder(tf.float32, shape=(args.batch_size, 32, 32, 3)))

    with tf.device('/gpu:%d' % i):

        # train
        gen_par = model(xs[i], ema=None, dropout_p=args.dropout_p, **model_opt)
        loss_gen.append(nn.discretized_mix_logistic_loss(xs[i], gen_par))

        # gradients
        grads.append(tf.gradients(loss_gen[i], all_params))

        # test
        gen_par = model(xs[i], ema=ema, dropout_p=0., **model_opt)
        loss_gen_test.append(nn.discretized_mix_logistic_loss(xs[i], gen_par))

# add gradients together and get training updates
tf_lr = tf.placeholder(tf.float32, shape=[])
with tf.device('/gpu:0'):
    for i in range(1,args.nr_gpu):
        loss_gen[0] += loss_gen[i]
        loss_gen_test[0] += loss_gen_test[i]
        for j in range(len(grads[0])):
            grads[0][j] += grads[i][j]

    # training ops
    optimizer = nn.adam_updates(all_params, grads[0], lr=tf_lr, mom1=0.95, mom2=0.9995)

# convert loss to bits / dim
bits_per_dim = loss_gen[0]/(args.nr_gpu*np.log(2.)*3*32*32*args.batch_size)
bits_per_dim_test = loss_gen_test[0]/(args.nr_gpu*np.log(2.)*3*32*32*args.batch_size)

# init & save
initializer = tf.initialize_all_variables()
saver = tf.train.Saver()

# load data
if not os.path.exists(args.data_dir):
    os.makedirs(args.data_dir)
if args.data_set == 'cifar':
    # load CIFAR-10 training data
    trainx, trainy = cifar10_data.load(args.data_dir + '/cifar-10-python')
    trainx = np.transpose(trainx, (0,2,3,1))
    nr_batches_train = int(trainx.shape[0]/args.batch_size)
    nr_batches_train_per_gpu = int(nr_batches_train/args.nr_gpu)

    # load CIFAR-10 test data
    testx, testy = cifar10_data.load(args.data_dir + '/cifar-10-python', subset='test')
    testx = np.transpose(testx, (0,2,3,1))
    nr_batches_test = int(testx.shape[0]/args.batch_size)
    nr_batches_test_per_gpu = int(nr_batches_test/args.nr_gpu)

elif args.data_set == 'imagenet':
    # download van Oord et al.'s small imagenet data set and convert using png_to_npz.py
    imgnet_data = np.load(args.data_dir + '/small_imagenet/imgnet_32x32.npz')
    trainx = imgnet_data['trainx']
    nr_batches_train = int(trainx.shape[0] / args.batch_size)
    nr_batches_train_per_gpu = int(nr_batches_train / args.nr_gpu)
    testx = imgnet_data['testx']
    nr_batches_test = int(testx.shape[0] / args.batch_size)
    nr_batches_test_per_gpu = int(nr_batches_test / args.nr_gpu)


# input to pixelCNN is scaled to [-1,1]
def scale_x(x):
    return np.cast[np.float32]((x - 127.5) / 127.5)

# //////////// perform training //////////////
if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)
print('starting training')
test_bpd = []
lr = args.learning_rate
with tf.Session() as sess:
    for epoch in range(5000):
        begin = time.time()

        # randomly permute
        inds = rng.permutation(trainx.shape[0])
        trainx = trainx[inds]

        # init
        if epoch==0:
            sess.run(initializer,{x_init: scale_x(trainx[:args.init_batch_size])})
            if args.load_params:
                saver.restore(sess, args.save_dir + '/params_' + args.data_set + '.ckpt')

        # train
        train_loss_gen = 0.
        for t in range(nr_batches_train_per_gpu):
            lr *= args.lr_decay
            feed_dict={tf_lr: lr}
            for i in range(args.nr_gpu):
                td =  t + i*nr_batches_train_per_gpu
                feed_dict[xs[i]] = scale_x(trainx[td * args.batch_size:(td + 1) * args.batch_size])
            l,_ = sess.run([bits_per_dim,optimizer], feed_dict)
            train_loss_gen += l
            sess.run(maintain_averages_op)
        train_loss_gen /= nr_batches_train_per_gpu

        # test
        test_loss_gen = 0.
        for t in range(nr_batches_test_per_gpu):
            feed_dict = {}
            for i in range(args.nr_gpu):
                td = t + i * nr_batches_test_per_gpu
                feed_dict[xs[i]] = scale_x(testx[td * args.batch_size:(td + 1) * args.batch_size])
            l = sess.run(bits_per_dim_test, feed_dict)
            test_loss_gen += l
        test_loss_gen /= nr_batches_test_per_gpu
        test_bpd.append(test_loss_gen)

        # log
        print("Iteration %d, time = %ds, train bits_per_dim = %.4f, test bits_per_dim = %.4f" % (epoch, time.time()-begin, train_loss_gen, test_loss_gen))
        sys.stdout.flush()

        if epoch % args.save_interval == 0:

            # generate samples from the model
            sample_x = sample_from_model(sess)
            img_tile = plotting.img_tile(sample_x, aspect_ratio=1.0, border_color=1.0, stretch=True)
            img = plotting.plot_img(img_tile, title='CIFAR10 samples')
            plotting.plt.savefig(args.save_dir + '/cifar10_sample' + str(epoch) + '.png')
            plotting.plt.close('all')

            # save params
            saver.save(sess, args.save_dir + '/params_' + args.data_set + '.ckpt')
            np.savez(args.save_dir + '/test_bpd_' + args.data_set + '.npz', test_bpd=np.array(test_bpd))

