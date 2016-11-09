"""
Various tensorflow utilities
"""

import numpy as np
import tensorflow as tf
from pixel_cnn_pp.scopes import add_arg_scope
from tensorflow.python.framework import function as tff

def int_shape(x):
    return list(map(int, x.get_shape()))

def concat_elu(x, axis=3):
    return tf.nn.elu(tf.concat(axis,[x,-x]))

def log_sum_exp(x):
    """ numerically stable log_sum_exp implementation that prevents overflow """
    axis = len(x.get_shape())-1
    m = tf.reduce_max(x, axis)
    m2 = tf.reduce_max(x, axis, keep_dims=True)
    return m + tf.log(tf.reduce_sum(tf.exp(x-m2), axis))

def log_prob_from_logits(x):
    """ numerically stable log_softmax implementation that prevents overflow """
    axis = len(x.get_shape())-1
    m = tf.reduce_max(x, axis, keep_dims=True)
    return x - m - tf.log(tf.reduce_sum(tf.exp(x-m), axis, keep_dims=True))

def discretized_mix_logistic_loss(x,l,sum_all=True):
    """ log-likelihood for mixture of discretized logistics, assumes the data has been rescaled to [-1,1] interval """
    xs = int_shape(x) # true image (i.e. labels) to regress to, e.g. (B,32,32,3)
    ls = int_shape(l) # predicted distribution, e.g. (B,32,32,100)
    nr_mix = int(ls[-1] / 10) # here and below: unpacking the params of the mixture of logistics
    logit_probs = l[:,:,:,:nr_mix]
    l = tf.reshape(l[:,:,:,nr_mix:], xs + [nr_mix*3])
    means = l[:,:,:,:,:nr_mix]
    log_scales = tf.maximum(l[:,:,:,:,nr_mix:2*nr_mix], -7.)
    coeffs = tf.nn.tanh(l[:,:,:,:,2*nr_mix:3*nr_mix])
    x = tf.reshape(x, xs + [1]) + tf.zeros(xs + [nr_mix]) # here and below: getting the means and adjusting them based on preceding sub-pixels
    m2 = tf.reshape(means[:,:,:,1,:] + coeffs[:, :, :, 0, :] * x[:, :, :, 0, :], [xs[0],xs[1],xs[2],1,nr_mix])
    m3 = tf.reshape(means[:, :, :, 2, :] + coeffs[:, :, :, 1, :] * x[:, :, :, 0, :] + coeffs[:, :, :, 2, :] * x[:, :, :, 1, :], [xs[0],xs[1],xs[2],1,nr_mix])
    means = tf.concat(3,[tf.reshape(means[:,:,:,0,:], [xs[0],xs[1],xs[2],1,nr_mix]), m2, m3])
    centered_x = x - means
    inv_stdv = tf.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1./255.)
    cdf_plus = tf.nn.sigmoid(plus_in)
    min_in = inv_stdv * (centered_x - 1./255.)
    cdf_min = tf.nn.sigmoid(min_in)
    log_cdf_plus = plus_in - tf.nn.softplus(plus_in) # log probability for edge case of 0 (before scaling)
    log_one_minus_cdf_min = -tf.nn.softplus(min_in) # log probability for edge case of 255 (before scaling)
    cdf_delta = cdf_plus - cdf_min # probability for all other cases
    mid_in = inv_stdv * centered_x
    log_pdf_mid = mid_in - log_scales - 2.*tf.nn.softplus(mid_in) # log probability in the center of the bin, to be used in extreme cases (not actually used in our code)

    # now select the right output: left edge case, right edge case, normal case, extremely low prob case (doesn't actually happen for us)

    # this is what we are really doing, but using the robust version below for extreme cases in other applications and to avoid NaN issue with tf.select()
    # log_probs = tf.select(x < -0.999, log_cdf_plus, tf.select(x > 0.999, log_one_minus_cdf_min, tf.log(cdf_delta)))

    # robust version, that still works if probabilities are below 1e-5 (which never happens in our code)
    # tensorflow backpropagates through tf.select() by multiplying with zero instead of selecting: this requires use to use some ugly tricks to avoid potential NaNs
    # the 1e-12 in tf.maximum(cdf_delta, 1e-12) is never actually used as output, it's purely there to get around the tf.select() gradient issue
    # if the probability on a sub-pixel is below 1e-5, we use an approximation based on the assumption that the log-density is constant in the bin of the observed sub-pixel value
    log_probs = tf.select(x < -0.999, log_cdf_plus, tf.select(x > 0.999, log_one_minus_cdf_min, tf.select(cdf_delta > 1e-5, tf.log(tf.maximum(cdf_delta, 1e-12)), log_pdf_mid - np.log(127.5))))

    log_probs = tf.reduce_sum(log_probs,3) + log_prob_from_logits(logit_probs)
    if sum_all:
        return -tf.reduce_sum(log_sum_exp(log_probs))
    else:
        return -tf.reduce_sum(log_sum_exp(log_probs),[1,2])

def sample_from_discretized_mix_logistic(l,nr_mix):
    ls = int_shape(l)
    xs = ls[:-1] + [3]
    # unpack parameters
    logit_probs = l[:, :, :, :nr_mix]
    l = tf.reshape(l[:, :, :, nr_mix:], xs + [nr_mix*3])
    # sample mixture indicator from softmax
    sel = tf.one_hot(tf.argmax(logit_probs - tf.log(-tf.log(tf.random_uniform(logit_probs.get_shape(), minval=1e-5, maxval=1. - 1e-5))), 3), depth=nr_mix, dtype=tf.float32)
    sel = tf.reshape(sel, xs[:-1] + [1,nr_mix])
    # select logistic parameters
    means = tf.reduce_sum(l[:,:,:,:,:nr_mix]*sel,4)
    log_scales = tf.maximum(tf.reduce_sum(l[:,:,:,:,nr_mix:2*nr_mix]*sel,4), -7.)
    coeffs = tf.reduce_sum(tf.nn.tanh(l[:,:,:,:,2*nr_mix:3*nr_mix])*sel,4)
    # sample from logistic & clip to interval
    # we don't actually round to the nearest 8bit value when sampling
    u = tf.random_uniform(means.get_shape(), minval=1e-5, maxval=1. - 1e-5)
    x = means + tf.exp(log_scales)*(tf.log(u) - tf.log(1. - u))
    x0 = tf.minimum(tf.maximum(x[:,:,:,0], -1.), 1.)
    x1 = tf.minimum(tf.maximum(x[:,:,:,1] + coeffs[:,:,:,0]*x0, -1.), 1.)
    x2 = tf.minimum(tf.maximum(x[:,:,:,2] + coeffs[:,:,:,1]*x0 + coeffs[:,:,:,2]*x1, -1.), 1.)
    return tf.concat(3,[tf.reshape(x0,xs[:-1]+[1]), tf.reshape(x1,xs[:-1]+[1]), tf.reshape(x2,xs[:-1]+[1])])

def get_var_maybe_avg(var_name, ema, **kwargs):
    ''' utility for retrieving polyak averaged params '''
    v = tf.get_variable(var_name, **kwargs)
    if ema is not None:
        v = ema.average(v)
    return v

def get_vars_maybe_avg(var_names, ema, **kwargs):
    ''' utility for retrieving polyak averaged params '''
    vars = []
    for vn in var_names:
        vars.append(get_var_maybe_avg(vn, ema, **kwargs))
    return vars

def adam_updates(params, cost_or_grads, lr=0.001, mom1=0.9, mom2=0.999):
    ''' Adam optimizer '''
    updates = []
    if type(cost_or_grads) is not list:
        grads = tf.gradients(cost_or_grads, params)
    else:
        grads = cost_or_grads
    t = tf.Variable(1., 'adam_t')
    for p, g in zip(params, grads):
        mg = tf.Variable(tf.zeros(p.get_shape()), p.name + '_adam_mg')
        if mom1>0:
            v = tf.Variable(tf.zeros(p.get_shape()), p.name + '_adam_v')
            v_t = mom1*v + (1. - mom1)*g
            v_hat = v_t / (1. - tf.pow(mom1,t))
            updates.append(v.assign(v_t))
        else:
            v_hat = g
        mg_t = mom2*mg + (1. - mom2)*tf.square(g)
        mg_hat = mg_t / (1. - tf.pow(mom2,t))
        g_t = v_hat / tf.sqrt(mg_hat + 1e-8)
        p_t = p - lr * g_t
        updates.append(mg.assign(mg_t))
        updates.append(p.assign(p_t))
    updates.append(t.assign_add(1))
    return tf.group(*updates)

def get_name(layer_name, counters):
    ''' utlity for keeping track of layer names '''
    if not layer_name in counters:
        counters[layer_name] = 0
    name = layer_name + '_' + str(counters[layer_name])
    counters[layer_name] += 1
    return name

# get params, using data based initialization & (optionally) weight normalization, and using moving averages
def get_params(layer_name, x=None, init=False, ema=None, use_W=True, use_g=True, use_b=True,
               f=tf.matmul, weight_norm=True, init_scale=1., filter_size=None, num_units=None):
    params = {}
    with tf.variable_scope(layer_name):
        if init:
            xs = int_shape(x)
            if num_units is None:
                num_units = xs[-1]
            norm_axes = [i for i in np.arange(len(xs) - 1)]

            # weights
            if use_W:
                if filter_size is not None:
                    V = tf.get_variable('V', filter_size + [xs[-1], num_units], tf.float32,
                                        tf.random_normal_initializer(0, 0.05), trainable=True)
                else:
                    V = tf.get_variable('V', [xs[-1], num_units], tf.float32,
                                    tf.random_normal_initializer(0, 0.05), trainable=True)
                if weight_norm:
                    W = tf.nn.l2_normalize(V.initialized_value(), [i for i in np.arange(len(V.get_shape())-1)])
                else:
                    W = V.initialized_value()

            # moments for normalization
            if use_W:
                x_init = f(x, W)
            else:
                x_init = x
            m_init, v_init = tf.nn.moments(x_init, norm_axes)

            # scale
            init_g = init_scale / tf.sqrt(v_init)
            if use_g:
                g = tf.get_variable('g', dtype=tf.float32, initializer=init_g, trainable=True).initialized_value()
                if use_W:
                    W *= tf.reshape(g, [1]*(len(W.get_shape())-1)+[num_units])
                else: # g is used directly if there are no weights
                    params['g'] = g
                m_init *= init_g
            elif use_W and not weight_norm: # init is the same as when using weight norm
                W = V.assign(tf.reshape(init_g, [1]*(len(W.get_shape())-1) + [num_units]) * W)
                m_init *= init_g

            # (possibly) scaled weights
            if use_W:
                params['W'] = W

            # bias
            if use_b:
                b = tf.get_variable('b', dtype=tf.float32, initializer=-m_init, trainable=True).initialized_value()
                params['b'] = b

        else:
            # get variables, use the exponential moving average if provided
            if use_b:
                params['b'] = get_var_maybe_avg('b', ema)
            if use_g:
                g = get_var_maybe_avg('g', ema)
                if not use_W: # g is used directly if there are no weights
                    params['g'] = g
            if use_W:
                V = get_var_maybe_avg('V', ema)
                Vs = int_shape(V)
                if weight_norm:
                    W = tf.nn.l2_normalize(V, [i for i in np.arange(len(Vs)-1)])
                else:
                    W = V
                if use_g:
                    W *= tf.reshape(g, [1]*(len(Vs)-1) + [Vs[-1]])
                params['W'] = W

    return params


''' utilities for shifting the image around, efficient alternative to masking convolutions '''

def down_shift(x):
    return tf.pad(x[:,:-1,:,:], [[0,0],[1,0],[0,0],[0,0]])

def right_shift(x):
    return tf.pad(x[:,:,:-1,:], [[0,0],[0,0],[1,0],[0,0]])

def up_shift(x):
    return tf.pad(x[:,1:,:,:], [[0,0],[0,1],[0,0],[0,0]])

def left_shift(x):
    return tf.pad(x[:,:,1:,:], [[0,0],[0,0],[0,1],[0,0]])


''' convolution / deconvolution wrappers supporting shifting the output: efficient alternative to masking '''

def _conv2d(x, W, stride, shift=None, filter_size=None):
    if filter_size is None:
        filter_size = int_shape(W)
    if shift is None:
        return tf.nn.conv2d(x, W, [1] + stride + [1], 'SAME')
    elif shift == 'down':
        x = tf.pad(x, [[0, 0], [filter_size[0] - 1, 0], [int((filter_size[1] - 1) / 2), int((filter_size[1] - 1) / 2)], [0, 0]])
    elif shift == 'down_right':
        x = tf.pad(x, [[0, 0], [filter_size[0] - 1, 0], [filter_size[1] - 1, 0], [0, 0]])
    elif shift == 'up':
        x = tf.pad(x, [[0, 0], [0, filter_size[0] - 1], [int((filter_size[1] - 1) / 2), int((filter_size[1] - 1) / 2)], [0, 0]])
    elif shift == 'up_left':
        x = tf.pad(x, [[0, 0], [0, filter_size[0] - 1], [0, filter_size[1] - 1], [0, 0]])
    else:
        raise('shift= ' + str(shift) + ' is a not supported')
    return tf.nn.conv2d(x, W, [1] + stride + [1], 'VALID')

def _deconv2d(x, W, stride, shift=None, xs=None, filter_size=None):
    if xs is None:
        xs = int_shape(x)
    if filter_size is None:
        filter_size = int_shape(W)
    num_filters = filter_size[-1]
    W_flipped = tf.transpose(W, [0,1,3,2])
    if shift is None:
        ts = [xs[0], xs[1] * stride[0], xs[2] * stride[1], num_filters]
        return tf.nn.conv2d_transpose(x, W_flipped, ts, [1] + stride + [1], padding='SAME')
    else:
        ts = [xs[0], xs[1] * stride[0] + filter_size[0] - 1, xs[2] * stride[1] + filter_size[1] - 1, num_filters]
        x = tf.nn.conv2d_transpose(x, W_flipped, ts, [1] + stride + [1], padding='VALID')

    if shift == 'down':
        return x[:,:(ts[1]-filter_size[0]+1),int((filter_size[1]-1)/2):(ts[2]-int((filter_size[1]-1)/2)),:]
    elif shift == 'down_right':
        return x[:,:(ts[1]-filter_size[0]+1),:(ts[2]-filter_size[1]+1),:]
    elif shift == 'up':
        return x[:, (filter_size[0] - 1):, int((filter_size[1] - 1) / 2):(ts[2] - int((filter_size[1] - 1) / 2)), :]
    elif shift == 'up_left':
        return x[:, (filter_size[0] - 1):, (filter_size[1] - 1):, :]
    else:
        raise ('shift= ' + str(shift) + ' is a not supported')


''' layer definitions '''

@add_arg_scope
def dense(x, num_units, nonlinearity=None, init_scale=1., counters={}, init=False,
          ema=None, weight_norm=True, use_b=True, use_g=True, **kwargs):
    layer_name = get_name('dense', counters)
    params = get_params(layer_name, x, init, ema, use_W=True, use_g=use_g, use_b=use_b,
                        f=tf.matmul, weight_norm=weight_norm, init_scale=init_scale, num_units=num_units)

    x = tf.matmul(x, params['W'])
    if use_b:
        x = tf.nn.bias_add(x, params['b'])
    if nonlinearity is not None:
        x = nonlinearity(x)
    return x

@add_arg_scope
def conv2d(x, num_filters, filter_size=[3,3], stride=[1,1], shift=None, nonlinearity=None, init_scale=1.,
           counters={}, init=False, ema=None, weight_norm=True, use_b=True, use_g=False, **kwargs):
    layer_name = get_name('conv2d', counters)
    f = lambda x,W: _conv2d(x, W, stride, shift)
    params = get_params(layer_name, x, init, ema, use_W=True, use_g=use_g, use_b=use_b,
                        f=f, weight_norm=weight_norm, init_scale=init_scale, filter_size=filter_size, num_units=num_filters)

    x = f(x, params['W'])
    if use_b:
        x = tf.nn.bias_add(x, params['b'])
    if nonlinearity is not None:
        x = nonlinearity(x)
    return x

@add_arg_scope
def deconv2d(x, num_filters, filter_size=[3,3], stride=[1,1], shift=None, nonlinearity=None, init_scale=1.,
             counters={}, init=False, ema=None, weight_norm=True, use_b=True, use_g=False,  **kwargs):
    layer_name = get_name('deconv2d', counters)
    f = lambda x, W: _deconv2d(x, W, stride, shift)
    params = get_params(layer_name, x, init, ema, use_W=True, use_g=use_g, use_b=use_b,
                        f=f, weight_norm=weight_norm, init_scale=init_scale, filter_size=filter_size, num_units=num_filters)

    x = f(x, params['W'])
    if use_b:
        x = tf.nn.bias_add(x, params['b'])
    if nonlinearity is not None:
        x = nonlinearity(x)
    return x

@add_arg_scope
def nin(x, num_units, **kwargs):
    """ a network in network layer (1x1 CONV) """
    sx = int_shape(x)
    x = tf.reshape(x, [-1, sx[-1]])
    x = dense(x, num_units, **kwargs)
    return tf.reshape(x, [-1] + sx[1:-1] + [num_units])


''' memory-efficient resnet layers '''
mem_funcs = {}

def my_bias_add(x, b, num_filters):
    return x + tf.reshape(b, [1, 1, 1, num_filters])

@add_arg_scope
def resnet(x, nonlinearity=concat_elu, shift=None, filter_size=[3,3], dropout_p=0.,
           save_memory=True, counters={}, init=False, ema=None, **kwargs):
    layer_name1 = get_name('resnet_first_conv2d', counters)
    layer_name2 = get_name('resnet_second_conv2d', counters)
    xs = int_shape(x)
    num_filters = int(xs[-1])
    f = lambda x, W: _conv2d(x, W, stride=[1, 1], shift=shift, filter_size=filter_size)

    if init:
        x_res = nonlinearity(x)
        params1 = get_params(layer_name1, x_res, init, ema, f=f, init_scale=1., filter_size=filter_size, num_units=num_filters)
        x_res = nonlinearity(tf.nn.bias_add(f(x_res, params1['W']), params1['b']))
        if dropout_p > 0:
            x_res = tf.nn.dropout(x_res, keep_prob=1. - dropout_p)
        params2 = get_params(layer_name2, x_res, init, ema, f=f, init_scale=0.1, filter_size=filter_size, num_units=num_filters)
        x_res = tf.nn.bias_add(f(x_res, params2['W']), params2['b'])
        return x + x_res

    else:

        f_name = (get_name('resnet_mem', counters) + str(xs)).replace(' ','_').replace('[','_').replace(']','_').replace(',','_')
        if f_name in mem_funcs:
            _resnet = mem_funcs[f_name]
        else:
            @tff.Defun(*([tf.float32] * 5), func_name=f_name)
            def _resnet(x, W1, b1, W2, b2):
                x_res = nonlinearity(my_bias_add(f(nonlinearity(x), W1), b1, num_filters))
                if dropout_p > 0:
                    x_res = tf.nn.dropout(x_res, keep_prob=1. - dropout_p)
                x_res = my_bias_add(f(x_res, W2), b2, num_filters)
                return x + x_res.set_shape(xs)

            mem_funcs[f_name] = _resnet

        params1 = get_params(layer_name1, ema=ema)
        params2 = get_params(layer_name2, ema=ema)
        y = _resnet(x, tf.convert_to_tensor(params1['W']), tf.convert_to_tensor(params1['b']),
                       tf.convert_to_tensor(params2['W']), tf.convert_to_tensor(params2['b']))
        y.set_shape(xs)
        return y

@add_arg_scope
def gated_resnet(x, nonlinearity=concat_elu, shift=None, filter_size=[3,3], dropout_p=0.,
           save_memory=True, counters={}, init=False, ema=None, **kwargs):
    layer_name1 = get_name('gated_resnet_first_conv2d', counters)
    layer_name2 = get_name('gated_resnet_second_conv2d', counters)
    xs = int_shape(x)
    num_filters = int(xs[-1])
    f = lambda x, W: _conv2d(x, W, stride=[1, 1], shift=shift, filter_size=filter_size)

    if init:
        x_res = nonlinearity(x)
        params1 = get_params(layer_name1, x_res, init, ema, f=f, init_scale=1., filter_size=filter_size, num_units=num_filters)
        x_res = nonlinearity(tf.nn.bias_add(f(x_res, params1['W']), params1['b']))
        if dropout_p > 0:
            x_res = tf.nn.dropout(x_res, keep_prob=1. - dropout_p)
        params2 = get_params(layer_name2, x_res, init, ema, f=f, init_scale=0.1, filter_size=filter_size, num_units=2*num_filters)
        x_res = tf.nn.bias_add(f(x_res, params2['W']), params2['b'])
        return x + tf.nn.sigmoid(x_res[:,:,:,:num_filters])*x_res[:,:,:,num_filters:]

    else:

        f_name = (get_name('resnet_mem', counters) + str(xs)).replace(' ', '_').replace('[', '_').replace(']', '_').replace(',','_')
        if f_name in mem_funcs:
            _resnet = mem_funcs[f_name]
        else:
            @tff.Defun(*([tf.float32] * 5), func_name=f_name)
            def _resnet(x, W1, b1, W2, b2):
                x_res = nonlinearity(my_bias_add(f(nonlinearity(x), W1), b1, num_filters))
                if dropout_p > 0:
                    x_res = tf.nn.dropout(x_res, keep_prob=1. - dropout_p)
                x_res = my_bias_add(f(x_res, W2), b2, 2*num_filters)
                a, b = tf.split(3, 2, x_res)
                return x + tf.nn.sigmoid(a) * b

            mem_funcs[f_name] = _resnet

        params1 = get_params(layer_name1, ema=ema)
        params2 = get_params(layer_name2, ema=ema)
        y = _resnet(x, tf.convert_to_tensor(params1['W']), tf.convert_to_tensor(params1['b']),
                       tf.convert_to_tensor(params2['W']), tf.convert_to_tensor(params2['b']))
        y.set_shape(xs)
        return y

def _one_by_one_conv(x, W, xs=None, num_filters=None):
    if xs is None:
        xs = int_shape(x)
    if num_filters is None:
        num_filters = int_shape(W)[-1]
    x = tf.reshape(x, [-1,xs[-1]])
    x = tf.matmul(x, W)
    return tf.reshape(x, [-1]+xs[1:-1]+[num_filters])

@add_arg_scope
def aux_gated_resnet(x, u, nonlinearity=concat_elu, shift=None, filter_size=[3,3], dropout_p=0.,
           save_memory=True, counters={}, init=False, ema=None, **kwargs):
    layer_name1 = get_name('aux_gated_resnet_nin', counters)
    layer_name2 = get_name('aux_gated_resnet_first_conv2d', counters)
    layer_name3 = get_name('aux_gated_resnet_second_conv2d', counters)
    xs = int_shape(x)
    us = int_shape(nonlinearity(u))
    num_filters = int(xs[-1])
    f = lambda x, W: _conv2d(x, W, stride=[1, 1], shift=shift, filter_size=filter_size)
    f_nin = lambda u, W: _one_by_one_conv(u, W, us, num_filters)

    if init:
        x_res = nonlinearity(x)
        u = nonlinearity(u)
        params1 = get_params(layer_name1, u, init, ema, f=f_nin, init_scale=1., num_units=num_filters, use_b=False)
        params2 = get_params(layer_name2, x_res, init, ema, f=f, init_scale=1., filter_size=filter_size, num_units=num_filters)
        x_res = nonlinearity(tf.nn.bias_add(f_nin(u, params1['W']) + f(x_res, params2['W']), params2['b']))
        if dropout_p > 0:
            x_res = tf.nn.dropout(x_res, keep_prob=1. - dropout_p)
        params3 = get_params(layer_name3, x_res, init, ema, f=f, init_scale=0.1, filter_size=filter_size, num_units=2*num_filters)
        x_res = tf.nn.bias_add(f(x_res, params3['W']), params3['b'])
        return x + tf.nn.sigmoid(x_res[:,:,:,:num_filters])*x_res[:,:,:,num_filters:]

    else:

        f_name = (get_name('resnet_mem', counters) + str(xs)).replace(' ', '_').replace('[', '_').replace(']', '_').replace(',','_')
        if f_name in mem_funcs:
            _resnet = mem_funcs[f_name]
        else:
            @tff.Defun(*([tf.float32] * 7), func_name=f_name)
            def _resnet(x, u, W1, W2, b2, W3, b3):
                x_res = nonlinearity(my_bias_add(f_nin(nonlinearity(u), W1) + f(nonlinearity(x), W2), b2, num_filters))
                if dropout_p > 0:
                    x_res = tf.nn.dropout(x_res, keep_prob=1. - dropout_p)
                x_res = my_bias_add(f(x_res, W3), b3, 2*num_filters)
                a, b = tf.split(3, 2, x_res)
                return x + tf.nn.sigmoid(a) * b

            mem_funcs[f_name] = _resnet

        params1 = get_params(layer_name1, ema=ema, use_b=False)
        params2 = get_params(layer_name2, ema=ema)
        params3 = get_params(layer_name3, ema=ema)
        y = _resnet(x, u, tf.convert_to_tensor(params1['W']), tf.convert_to_tensor(params2['W']),
                       tf.convert_to_tensor(params2['b']), tf.convert_to_tensor(params3['W']), tf.convert_to_tensor(params3['b']))
        y.set_shape(xs)
        return y
