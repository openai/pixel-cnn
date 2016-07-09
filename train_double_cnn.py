import argparse
import time
import sys
import os
import numpy as np
import tensorflow as tf
import scopes
import nn
import plotting
import cifar10_data
sys.setrecursionlimit(10000)

# settings
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--init_batch_size', type=int, default=256)
parser.add_argument('--sample_batch_size', type=int, default=64)
parser.add_argument('--nr_resnet', type=int, default=5)
parser.add_argument('--nr_logistic_mix', type=int, default=10)
parser.add_argument('--nr_gpu', type=int, default=8)
parser.add_argument('--learning_rate', type=float, default=0.003)
parser.add_argument('--nr_filters', type=int, default=64)
args = parser.parse_args()
print(args)

# fix random seed
rng = np.random.RandomState(args.seed)

# conditioning on pixels above and to the left
def conditioner_spec(x, init=False, ema=None):
    counters = {}
    with scopes.arg_scope([nn.down_shifted_conv2d, nn.down_right_shifted_conv2d, nn.down_shifted_deconv2d, nn.down_right_shifted_deconv2d, nn.nin],
                          counters=counters, init=init, ema=ema):

        # ////////// up pass ////////
        xs = nn.int_shape(x)
        x_pad = tf.concat(3,[x,tf.ones(xs[:-1]+[1])]) # add channel of ones to distinguish image from padding later on
        u_list = [nn.down_shifted_conv2d(x_pad, num_filters=args.nr_filters, filter_size=[2, 3])] # stream for current row + up
        ul_list = [nn.down_shift(nn.down_shifted_conv2d(x_pad, num_filters=args.nr_filters, filter_size=[1,3])) + \
                   nn.right_shift(nn.down_right_shifted_conv2d(x, num_filters=args.nr_filters, filter_size=[2,1]))] # stream for up and to the left

        for rep in range(args.nr_resnet):
            u_list.append(nn.gated_resnet(u_list[-1], conv=nn.down_shifted_conv2d))
            ul_list.append(nn.aux_gated_resnet(ul_list[-1], nn.down_shift(u_list[-1]), conv=nn.down_right_shifted_conv2d))

        u_list.append(nn.down_shifted_conv2d(u_list[-1], num_filters=args.nr_filters, stride=[2, 2]))
        ul_list.append(nn.down_right_shifted_conv2d(ul_list[-1], num_filters=args.nr_filters, stride=[2, 2]))

        for rep in range(args.nr_resnet):
            u_list.append(nn.gated_resnet(u_list[-1], conv=nn.down_shifted_conv2d))
            ul_list.append(nn.aux_gated_resnet(ul_list[-1], nn.down_shift(u_list[-1]), conv=nn.down_right_shifted_conv2d))

        u_list.append(nn.down_shifted_conv2d(u_list[-1], num_filters=args.nr_filters, stride=[2, 2]))
        ul_list.append(nn.down_right_shifted_conv2d(ul_list[-1], num_filters=args.nr_filters, stride=[2, 2]))

        for rep in range(args.nr_resnet):
            u_list.append(nn.gated_resnet(u_list[-1], conv=nn.down_shifted_conv2d))
            ul_list.append(nn.aux_gated_resnet(ul_list[-1], nn.down_shift(u_list[-1]), conv=nn.down_right_shifted_conv2d))

        # /////// down pass ////////
        u = u_list.pop()
        ul = ul_list.pop()

        for rep in range(args.nr_resnet):
            u = nn.aux_gated_resnet(u, u_list.pop(), conv=nn.down_shifted_conv2d)
            ul = nn.aux_gated_resnet(ul, tf.concat(3,[nn.down_shift(u), ul_list.pop()]), conv=nn.down_right_shifted_conv2d)

        u = nn.down_shifted_deconv2d(u, num_filters=args.nr_filters, stride=[2, 2])
        ul = nn.down_right_shifted_deconv2d(ul, num_filters=args.nr_filters, stride=[2, 2])

        for rep in range(args.nr_resnet+1):
            u = nn.aux_gated_resnet(u, u_list.pop(), conv=nn.down_shifted_conv2d)
            ul = nn.aux_gated_resnet(ul, tf.concat(3, [nn.down_shift(u), ul_list.pop()]), conv=nn.down_right_shifted_conv2d)

        u = nn.down_shifted_deconv2d(u, num_filters=args.nr_filters, stride=[2, 2])
        ul = nn.down_right_shifted_deconv2d(ul, num_filters=args.nr_filters, stride=[2, 2])

        for rep in range(args.nr_resnet+1):
            u = nn.aux_gated_resnet(u, u_list.pop(), conv=nn.down_shifted_conv2d)
            ul = nn.aux_gated_resnet(ul, tf.concat(3, [nn.down_shift(u), ul_list.pop()]), conv=nn.down_right_shifted_conv2d)

        x_out = nn.nin(nn.concat_elu(ul),10*args.nr_logistic_mix)

        assert len(u_list) == 0
        assert len(ul_list) == 0

    return x_out
conditioner = tf.make_template('conditioner',conditioner_spec)

# data
x_init = tf.placeholder(tf.float32, shape=(args.init_batch_size, 32, 32, 3))

# run once for data dependent initialization of parameters
conditioner(x_init, init=True)

# get list of all params
all_params = tf.trainable_variables()

# keep track of moving average
ema = tf.train.ExponentialMovingAverage(decay=0.999)
maintain_averages_op = tf.group(ema.apply(all_params))

# sample from the model
x_sample = tf.placeholder(tf.float32, shape=(args.sample_batch_size, 32, 32, 3))
new_x_gen = nn.sample_from_discretized_mix_logistic(conditioner(x_sample, ema=ema), args.nr_logistic_mix)
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
loss = []
loss_test = []
for i in range(args.nr_gpu):
    xs.append(tf.placeholder(tf.float32, shape=(args.batch_size, 32, 32, 3)))

    with tf.device('/gpu:%d' % i):

        # train
        loss.append(-nn.discretized_mix_logistic(xs[i], conditioner(xs[i])))

        # gradients
        grads.append(tf.gradients(loss[i],all_params))

        # test
        loss_test.append(-nn.discretized_mix_logistic(xs[i], conditioner(xs[i], ema=ema)))

# add gradients together and get training updates
with tf.device('/gpu:0'):
    for i in range(1,args.nr_gpu):
        loss[0] += loss[i]
        loss_test[0] += loss_test[i]
        for j in range(len(grads[0])):
            grads[0][j] += grads[i][j]

    # training ops
    optimizer = nn.adamax_updates(all_params, grads[0], lr=args.learning_rate)

# convert loss to bits / dim
bits_per_dim = loss[0]/(args.nr_gpu*np.log(2.)*3*32*32*args.batch_size)
bits_per_dim_test = loss_test[0]/(args.nr_gpu*np.log(2.)*3*32*32*args.batch_size)

# init & save
initializer = tf.initialize_all_variables()
saver = tf.train.Saver()

# load CIFAR-10 training data
trainx, _ = cifar10_data.load('/home/tim/data/cifar-10-python')
trainx = np.transpose(trainx, (0,2,3,1))
nr_batches_train = int(trainx.shape[0]/args.batch_size)
nr_batches_train_per_gpu = nr_batches_train/args.nr_gpu

# load CIFAR-10 test data
testx, _ = cifar10_data.load('/home/tim/data/cifar-10-python', subset='test')
testx = np.transpose(testx, (0,2,3,1))
nr_batches_test = int(testx.shape[0]/args.batch_size)
nr_batches_test_per_gpu = nr_batches_test/args.nr_gpu

# //////////// perform training //////////////
if not os.path.exists('/local_home/tim/pixel_cnn'):
    os.makedirs('/local_home/tim/pixel_cnn')
print('starting training')
begin_all = time.time()
with tf.Session() as sess:
    for epoch in range(10000):
        begin = time.time()

        # randomly permute
        trainx = trainx[rng.permutation(trainx.shape[0])]

        # init
        if epoch==0:
            sess.run(initializer,{x_init: trainx[:args.init_batch_size]})
            #saver.restore(sess, '/local_home/tim/pixel_cnn/params.ckpt')

        # train
        train_loss = 0.
        for t in range(nr_batches_train_per_gpu):
            feed_dict={}
            for i in range(args.nr_gpu):
                td =  t + i*nr_batches_train_per_gpu
                feed_dict[xs[i]] = trainx[td*args.batch_size:(td+1)*args.batch_size]
            l,_ = sess.run([bits_per_dim,optimizer], feed_dict)
            train_loss += l
            sess.run(maintain_averages_op)
        train_loss /= nr_batches_train_per_gpu

        # test
        test_loss = 0.
        for t in range(nr_batches_test_per_gpu):
            feed_dict = {}
            for i in range(args.nr_gpu):
                td = t + i * nr_batches_test_per_gpu
                feed_dict[xs[i]] = testx[td * args.batch_size:(td + 1) * args.batch_size]
            l = sess.run(bits_per_dim_test, feed_dict)
            test_loss += l
        test_loss /= nr_batches_test_per_gpu

        # log
        print("Iteration %d, time = %ds, train bits_per_dim = %.4f, test bits_per_dim = %.4f" % (epoch, time.time()-begin, train_loss, test_loss))
        sys.stdout.flush()

        if epoch%10 == 0:

            # generate samples from the model
            sample_x = sample_from_model(sess)
            img_tile = plotting.img_tile(sample_x, aspect_ratio=1.0, border_color=1.0, stretch=True)
            img = plotting.plot_img(img_tile, title='CIFAR10 samples')
            plotting.plt.savefig('/local_home/tim/pixel_cnn/cifar10_sample' + str(epoch) + '.png')
            plotting.plt.close('all')

            # save params
            saver.save(sess, '/local_home/tim/pixel_cnn/params' + str(epoch) + '.ckpt')

