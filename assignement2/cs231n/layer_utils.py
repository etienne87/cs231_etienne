from cs231n.layers import *
from cs231n.fast_layers import *

def affine_xxx_forward(x, params):
  bn_fwd = getattr(cs231n.layers,params['bn_rule']+'_forward')
  nl_fwd = getattr(cs231n.layers,params['nonlinearity']+'_forward')
  gamma, beta, bn_param = params.get('bn_params')

  a, fc_cache = affine_forward(x, w, b)
  an, bn_cache = bn_fwd(a, gamma, beta, bn_param )
  out, relu_cache = nl_fwd(an)
  cache = (fc_cache, bn_cache, relu_cache)
  return out, cache

def affine_xxx_backward(dout, cache, params):
  bn_bwd = getattr(cs231n.layers,params['bn_rule']+'_backward')
  nl_bwd = getattr(cs231n.layers,params['nonlinearity'+'_backward'])
  gamma, beta, bn_param = params.get('bn_params')

  
  fc_cache, bn_cache, relu_cache = cache
  da = relu_backward(dout, relu_cache)
  dan, dgamma, dbeta = batchnorm_backward(da, bn_cache)
  dx, dw, db = affine_backward(dan, fc_cache)
  return dx, dw, db, dgamma, dbeta



def affine_relu_forward(x, w, b):
  """
  Convenience layer that perorms an affine transform followed by a ReLU

  Inputs:
  - x: Input to the affine layer
  - w, b: Weights for the affine layer

  Returns a tuple of:
  - out: Output from the ReLU
  - cache: Object to give to the backward pass
  """
  a, fc_cache = affine_forward(x, w, b)
  out, relu_cache = relu_forward(a)
  cache = (fc_cache, relu_cache)
  return out, cache


def affine_relu_backward(dout, cache):
  """
  Backward pass for the affine-relu convenience layer
  """
  fc_cache, relu_cache = cache
  da = relu_backward(dout, relu_cache)
  dx, dw, db = affine_backward(da, fc_cache)
  return dx, dw, db


def affine_relu_bn_forward(x, w, b, gamma, beta, bn_param):
  """
  Convenience layer that perorms an affine transform followed by a Batch Normalization & ReLU

  Inputs:
  - x: Input to the affine layer
  - w, b: Weights for the affine layer

  Returns a tuple of:
  - out: Output from the ReLU
  - cache: Object to give to the backward pass
  """
  a, fc_cache = affine_forward(x, w, b)
  an, bn_cache = batchnorm_forward(a, gamma, beta, bn_param )
  out, relu_cache = relu_forward(an)
  cache = (fc_cache, bn_cache, relu_cache)
  return out, cache


def affine_relu_bn_backward(dout, cache):
  """
  Backward pass for the affine-relu convenience layer
  """
  fc_cache, bn_cache, relu_cache = cache
  da = relu_backward(dout, relu_cache)
  dan, dgamma, dbeta = batchnorm_backward(da, bn_cache)
  dx, dw, db = affine_backward(dan, fc_cache)
  return dx, dw, db, dgamma, dbeta


def affine_relu_rbn_forward(x, w, b, gamma, beta, bn_param):
  """
  Convenience layer that perorms an affine transform followed by a Batch Normalization & ReLU

  Inputs:
  - x: Input to the affine layer
  - w, b: Weights for the affine layer

  Returns a tuple of:
  - out: Output from the ReLU
  - cache: Object to give to the backward pass
  """
  a, fc_cache = affine_forward(x, w, b)
  an, bn_cache = running_batchnorm_forward(a, gamma, beta, bn_param )
  out, relu_cache = relu_forward(an)
  cache = (fc_cache, bn_cache, relu_cache)
  return out, cache


def affine_relu_rbn_backward(dout, cache):
  """
  Backward pass for the affine-relu convenience layer
  """
  fc_cache, bn_cache, relu_cache = cache
  da = relu_backward(dout, relu_cache)
  dan, dgamma, dbeta = running_batchnorm_backward(da, bn_cache)
  dx, dw, db = affine_backward(dan, fc_cache)
  return dx, dw, db, dgamma, dbeta

def affine_relu_mobn_forward(x, w, b, gamma, beta, bn_param):
  """
  Convenience layer that perorms an affine transform followed by a Batch Normalization & ReLU

  Inputs:
  - x: Input to the affine layer
  - w, b: Weights for the affine layer

  Returns a tuple of:
  - out: Output from the ReLU
  - cache: Object to give to the backward pass
  """
  a, fc_cache = affine_forward(x, w, b)
  an, bn_cache = meanonly_batchnorm_forward(a, gamma, beta, bn_param )
  out, relu_cache = relu_forward(an)
  cache = (fc_cache, bn_cache, relu_cache)
  return out, cache


def affine_relu_mobn_backward(dout, cache):
  """
  Backward pass for the affine-relu convenience layer
  """
  fc_cache, bn_cache, relu_cache = cache
  da = relu_backward(dout, relu_cache)
  dan, dgamma, dbeta = meanonly_batchnorm_backward(da, bn_cache)
  dx, dw, db = affine_backward(dan, fc_cache)
  return dx, dw, db, dgamma, dbeta



def affine_elu_bn_forward(x, w, b, gamma, beta, bn_param):
  """
  Convenience layer that perorms an affine transform followed by a Batch Normalization & ReLU

  Inputs:
  - x: Input to the affine layer
  - w, b: Weights for the affine layer

  Returns a tuple of:
  - out: Output from the ReLU
  - cache: Object to give to the backward pass
  """
  a, fc_cache = affine_forward(x, w, b)
  an, bn_cache = batchnorm_forward(a, gamma, beta, bn_param )
  out, relu_cache = elu_forward(an)
  cache = (fc_cache, bn_cache, relu_cache)
  return out, cache


def affine_elu_bn_backward(dout, cache):
  """
  Backward pass for the affine-relu convenience layer
  """
  fc_cache, bn_cache, relu_cache = cache
  da = elu_backward(dout, relu_cache)
  dan, dgamma, dbeta = batchnorm_backward(da, bn_cache)
  dx, dw, db = affine_backward(dan, fc_cache)
  return dx, dw, db, dgamma, dbeta


def affine_elu_rbn_forward(x, w, b, gamma, beta, bn_param):
  """
  Convenience layer that perorms an affine transform followed by a Batch Normalization & ReLU

  Inputs:
  - x: Input to the affine layer
  - w, b: Weights for the affine layer

  Returns a tuple of:
  - out: Output from the ReLU
  - cache: Object to give to the backward pass
  """
  a, fc_cache = affine_forward(x, w, b)
  an, bn_cache = running_batchnorm_forward(a, gamma, beta, bn_param )
  out, relu_cache = elu_forward(an)
  cache = (fc_cache, bn_cache, relu_cache)
  return out, cache


def affine_elu_rbn_backward(dout, cache):
  """
  Backward pass for the affine-relu convenience layer
  """
  fc_cache, bn_cache, relu_cache = cache
  da = elu_backward(dout, relu_cache)
  dan, dgamma, dbeta = running_batchnorm_backward(da, bn_cache)
  dx, dw, db = affine_backward(dan, fc_cache)
  return dx, dw, db, dgamma, dbeta


pass


def conv_relu_forward(x, w, b, conv_param):
  """
  A convenience layer that performs a convolution followed by a ReLU.

  Inputs:
  - x: Input to the convolutional layer
  - w, b, conv_param: Weights and parameters for the convolutional layer
  
  Returns a tuple of:
  - out: Output from the ReLU
  - cache: Object to give to the backward pass
  """
  a, conv_cache = conv_forward_fast(x, w, b, conv_param)
  out, relu_cache = relu_forward(a)
  cache = (conv_cache, relu_cache)
  return out, cache


def conv_relu_backward(dout, cache):
  """
  Backward pass for the conv-relu convenience layer.
  """
  conv_cache, relu_cache = cache
  da = relu_backward(dout, relu_cache)
  dx, dw, db = conv_backward_fast(da, conv_cache)
  return dx, dw, db


def conv_relu_pool_forward(x, w, b, conv_param, pool_param):
  """
  Convenience layer that performs a convolution, a ReLU, and a pool.

  Inputs:
  - x: Input to the convolutional layer
  - w, b, conv_param: Weights and parameters for the convolutional layer
  - pool_param: Parameters for the pooling layer

  Returns a tuple of:
  - out: Output from the pooling layer
  - cache: Object to give to the backward pass
  """
  a, conv_cache = conv_forward_fast(x, w, b, conv_param)
  s, relu_cache = relu_forward(a)
  out, pool_cache = max_pool_forward_fast(s, pool_param)
  cache = (conv_cache, relu_cache, pool_cache)
  return out, cache


def conv_relu_pool_backward(dout, cache):
  """
  Backward pass for the conv-relu-pool convenience layer
  """
  conv_cache, relu_cache, pool_cache = cache
  ds = max_pool_backward_fast(dout, pool_cache)
  da = relu_backward(ds, relu_cache)
  dx, dw, db = conv_backward_fast(da, conv_cache)
  return dx, dw, db

def conv_bn_relu_pool_forward(x, w, b, gamma, beta, conv_param, pool_param, bn_param):
  """
  Convenience layer that performs a convolution, a BN, a ReLU, and a pool.

  Inputs:
  - x: Input to the convolutional layer
  - w, b, conv_param: Weights and parameters for the convolutional layer
  - pool_param: Parameters for the pooling layer

  Returns a tuple of:
  - out: Output from the pooling layer
  - cache: Object to give to the backward pass
  """
  a, conv_cache = conv_forward_fast(x, w, b, conv_param)
  an, bn_cache = spatial_batchnorm_forward(a, gamma, beta, bn_param)
  s, relu_cache = relu_forward(a)
  out, pool_cache = max_pool_forward_fast(s, pool_param)
  cache = (conv_cache, relu_cache, bn_cache, pool_cache)
  return out, cache


def conv_bn_relu_pool_backward(dout, cache):
  """
  Backward pass for the conv-bn-relu-pool convenience layer
  """
  conv_cache, relu_cache, bn_cache, pool_cache = cache
  ds = max_pool_backward_fast(dout, pool_cache)
  da = relu_backward(ds, relu_cache)
  dan, dgamma, dbeta = spatial_batchnorm_backward(da, bn_cache)
  dx, dw, db = conv_backward_fast(da, conv_cache)
  return dx, dw, db, dgamma, dbeta


