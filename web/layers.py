import numpy as np


def affine_forward(x, w, b):
    x_re = x.reshape(x.shape[0],np.prod(x.shape[1:]))
    out = x_re.dot(w) + b
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    x, w, b = cache
    dx, dw, db = None, None, None
    db = np.sum(dout, axis=0).T
    x_re = x.reshape(x.shape[0],np.prod(x.shape[1:]))
    dw = x_re.T.dot(dout)
    dxND = dout.dot(w.T)
    dx = dxND.reshape(x.shape)
    return dx, dw, db


def relu_forward(x):
    out = np.maximum(0,x)
    cache = x
    return out, cache


def relu_backward(dout, cache):
    dx = dout * (cache>0)
    return dx


def softmax_loss(x, y):
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx


def affine_relu_forward(x, w, b):
    a, fc_cache = affine_forward(x, w, b)
    out, relu_cache = relu_forward(a)
    cache = (fc_cache, relu_cache)
    return out, cache


def affine_relu_backward(dout, cache):
    fc_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dw, db = affine_backward(da, fc_cache)
    return dx, dw, db