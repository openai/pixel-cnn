"""
The core Pixel-CNN model
"""

import tensorflow as tf

import pixel_cnn_pp.scopes as scopes
import pixel_cnn_pp.nn as nn

def model_spec(x, init=False, ema=None, dropout_p=0.5, nr_resnet=5, nr_filters=128, nr_logistic_mix=5, save_memory=True):
    """
    We receive a Tensor x of shape (N,H,W,D1) (e.g. (12,32,32,3)) and produce
    a Tensor x_out of shape (N,H,W,D2) (e.g. (12,32,32,100)), where each fiber
    of the x_out tensor describes the predictive distribution for the RGB at
    that position.
    """

    counters = {}
    with scopes.arg_scope([nn.conv2d, nn.deconv2d, nn.gated_resnet, nn.aux_gated_resnet, nn.dense],
                          counters=counters, init=init, ema=ema, dropout_p=dropout_p, save_memory=save_memory):

        # ////////// up pass through pixelCNN ////////
        xs = nn.int_shape(x)
        x_pad = tf.concat(3,[x,tf.ones(xs[:-1]+[1])]) # add channel of ones to distinguish image from padding later on
        u_list = [nn.down_shift(nn.conv2d(x_pad, num_filters=nr_filters, filter_size=[2, 3], shift='down'))] # stream for pixels above
        ul_list = [nn.down_shift(nn.conv2d(x_pad, num_filters=nr_filters, filter_size=[1,3])) + \
                   nn.right_shift(nn.conv2d(x_pad, num_filters=nr_filters, filter_size=[2,1], shift='down'))] # stream for up and to the left
        
        for rep in range(nr_resnet):
            u_list.append(nn.gated_resnet(u_list[-1], shift='down', filter_size=[2,3]))
            ul_list.append(nn.aux_gated_resnet(ul_list[-1], u_list[-1], shift='down_right', filter_size=[2,2]))
        
        u_list.append(nn.conv2d(u_list[-1], num_filters=nr_filters, stride=[2, 2], shift='down', filter_size=[2,3]))
        ul_list.append(nn.conv2d(ul_list[-1], num_filters=nr_filters, stride=[2, 2], shift='down_right', filter_size=[2,2]))

        for rep in range(nr_resnet):
            u_list.append(nn.gated_resnet(u_list[-1], shift='down', filter_size=[2,3]))
            ul_list.append(nn.aux_gated_resnet(ul_list[-1], u_list[-1], shift='down_right', filter_size=[2,2]))

        u_list.append(nn.conv2d(u_list[-1], num_filters=nr_filters, stride=[2, 2], shift='down', filter_size=[2,3]))
        ul_list.append(nn.conv2d(ul_list[-1], num_filters=nr_filters, stride=[2, 2], shift='down_right', filter_size=[2,2]))

        for rep in range(nr_resnet):
            u_list.append(nn.gated_resnet(u_list[-1], shift='down', filter_size=[2,3]))
            ul_list.append(nn.aux_gated_resnet(ul_list[-1], u_list[-1], shift='down_right', filter_size=[2,2]))

        # /////// down pass ////////
        u = u_list.pop()
        ul = ul_list.pop()
        for rep in range(nr_resnet):
            u = nn.aux_gated_resnet(u, u_list.pop(), shift='down', filter_size=[2,3])
            ul = nn.aux_gated_resnet(ul, tf.concat(3, [u, ul_list.pop()]), shift='down_right', filter_size=[2,2])

        u = nn.deconv2d(u, num_filters=nr_filters, stride=[2, 2], shift='down', filter_size=[2,3])
        ul = nn.deconv2d(ul, num_filters=nr_filters, stride=[2, 2], shift='down_right', filter_size=[2,2])

        for rep in range(nr_resnet+1):
            u = nn.aux_gated_resnet(u, u_list.pop(), shift='down', filter_size=[2,3])
            ul = nn.aux_gated_resnet(ul, tf.concat(3, [u, ul_list.pop()]), shift='down_right', filter_size=[2,2])

        u = nn.deconv2d(u, num_filters=nr_filters, stride=[2, 2], shift='down', filter_size=[2,3])
        ul = nn.deconv2d(ul, num_filters=nr_filters, stride=[2, 2], shift='down_right', filter_size=[2,2])

        for rep in range(nr_resnet+1):
            u = nn.aux_gated_resnet(u, u_list.pop(), shift='down', filter_size=[2,3])
            ul = nn.aux_gated_resnet(ul, tf.concat(3, [u, ul_list.pop()]), shift='down_right', filter_size=[2,2])

        x_out = nn.nin(nn.concat_elu(ul), 10*nr_logistic_mix)

        assert len(u_list) == 0
        assert len(ul_list) == 0

        return x_out

