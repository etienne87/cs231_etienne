import numpy as np


def affine_forward(x, w, b):
  """
  Computes the forward pass for an affine (fully-connected) layer.

  The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
  examples, where each example x[i] has shape (d_1, ..., d_k). We will
  reshape each input into a vector of dimension D = d_1 * ... * d_k, and
  then transform it to an output vector of dimension M.

  Inputs:
  - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
  - w: A numpy array of weights, of shape (D, M)
  - b: A numpy array of biases, of shape (M,)
  
  Returns a tuple of:
  - out: output, of shape (N, M)
  - cache: (x, w, b)
  """
  out = None
  #############################################################################
  # TODO: Implement the affine forward pass. Store the result in out. You     #
  # will need to reshape the input into rows.                                 #
  #############################################################################
  n = x.shape[0]
  xr = x.reshape((n,-1))
  out = xr.dot(w) + b
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, w, b)
  return out, cache


def affine_backward(dout, cache):
  """
  Computes the backward pass for an affine layer.

  Inputs:
  - dout: Upstream derivative, of shape (N, M)
  - cache: Tuple of:
    - x: Input data, of shape (N, d_1, ... d_k)
    - w: Weights, of shape (D, M)

  Returns a tuple of:
  - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
  - dw: Gradient with respect to w, of shape (D, M)
  - db: Gradient with respect to b, of shape (M,)
  """
  x, w, b = cache
  dx, dw, db = None, None, None
  #############################################################################
  # TODO: Implement the affine backward pass.                                 #
  #############################################################################
  xs = x.shape
  ndim = len(xs)
  n = xs[0]
  xr = x.reshape((n,-1))
  dx = (dout.dot(w.T)).reshape((xs))
  dw = xr.T.dot(dout)
  db = np.sum(dout,axis=0)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx, dw, db


def relu_forward(x):
  """
  Computes the forward pass for a layer of rectified linear units (ReLUs).

  Input:
  - x: Inputs, of any shape

  Returns a tuple of:
  - out: Output, of the same shape as x
  - cache: x
  """
  out = None
  #############################################################################
  # TODO: Implement the ReLU forward pass.                                    #
  #############################################################################
  out = np.maximum(0,x)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = x
  return out, cache


def relu_backward(dout, cache):
  """
  Computes the backward pass for a layer of rectified linear units (ReLUs).

  Input:
  - dout: Upstream derivatives, of any shape
  - cache: Input x, of same shape as dout

  Returns:
  - dx: Gradient with respect to x
  """
  dx, x = None, cache
  x = cache
  #############################################################################
  # TODO: Implement the ReLU backward pass.                                   #
  #############################################################################
  dx = dout
  dx[np.where(x<=0)] = 0
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx

def elu_forward(x):
    alpha = 1.0
    neg = np.where(x<=0)
    pos = np.where(x>0)
    out = np.zeros_like(x)
    out[pos] = x[pos]
    out[neg] = alpha * (np.exp(x[neg])-1)
    cache = x, out, pos, neg, alpha
    return out, cache

def elu_backward(dout, cache):
    x, out, pos, neg, alpha = cache
    dx = np.zeros_like(dout)
    dx[pos] = dout[pos]
    dx[neg] = dout[neg] * (out[neg]+alpha)
    return dx

from numpy import linalg as LA
def weightnorm_forward(g, v):
    """
    Weight Normalization :
    
    w = g * v / ||v|| ; g is scalar, v is k-dimension vector

    see http://arxiv.org/pdf/1602.07868v2.pdf

    todo : handle the case where v is multidim
    """
    if(v.ndim > 2):
        v1 = v.reshape(v.shape[0],-1).T
    else:
        v1 = v

    vnorm = LA.norm(v1, axis=0)
    vn = v1 / vnorm
    w = g * vn
    cache = (g,v1,vnorm)
    w = w.reshape(v.shape)
    return w, cache

def weightnorm_backward(dw, cache):
    """
    dg = dw * v / ||v||
    dv = g * dw / ||v|| - dg * v / ||v||² 
    """
    if(dw.ndim > 2):
        dw1 = dw.reshape(dw.shape[0],-1).T
    else:
        dw1 = dw

    g, v1, vnorm = cache
    vn = v1/(vnorm)
    dg = g * dw1 * vn
    dv = g * dw1 / (vnorm) - dg * v1 / (vnorm**2)
    dv = dv.reshape(dw.shape)
    return dg, dv

#  for fun, what if we complexify weightnorm pipeline a little bit?
#  v' =  v / ||v||     normalize
#  v'' = max( Th, v')  sparsify
#  w = g * v''         scale
def sparse_weightnorm_forward(g, v, theta = 0.01):
    vnorm = LA.norm(v, axis=0)
    vn = v / vnorm
    dabs = np.sign(vn)
    mask = ((vn*dabs)>=theta)*1
    vn2 = vn * mask
    w = g * v / vnorm
    cache = (g,v,vnorm,mask,dabs)
    return w, cache

def sparse_weightnorm_backward(dw, cache):
    g,v,vnorm,mask,dabs = cache
    vn = v/(vnorm)
    vn2 = vn * mask
    dg = g * dw * vn2
    dv = g * dw * (vn * mask * dabs + np.abs(vn) * mask) * (1.0 - vn**2) / vnorm
    return dg, dv


"""
mean-only batchnorm
"""
def meanonly_batchnorm_forward(x, gamma, beta, bn_param):
  mode = bn_param['mode']
  eps = bn_param.get('eps',0)
  momentum = bn_param.get('momentum', 0.9)

  N, D = x.shape
  running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))

  out, cache = None, None
  if mode == 'train':
    mean = np.mean(x, 0)
    out = (x-mean)
    cache = (out, mean)
    running_mean = momentum * running_mean + (1 - momentum) * mean
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
  elif mode == 'test':
    out = (x-bn_param['running_mean'])
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
  else:
    raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

  # Store the updated running means back into bn_param
  bn_param['running_mean'] = running_mean

  return out, cache

def meanonly_batchnorm_backward(dout, cache):
  dx, dgamma, dbeta = None, None, None
  #############################################################################
  # TODO: Implement the backward pass for batch normalization. Store the      #
  # results in the dx, dgamma, and dbeta variables.                           #
  #############################################################################
  xn, mean = cache
  m = xn.shape[0]
  dxn = dout 
  dmu = -np.sum(dxn,0)                     
  dx = dxn + dmu/m

  dgamma = np.zeros((1,xn.shape[1]))
  dbeta = np.zeros((xn.shape[0],1)) 
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return dx, dgamma, dbeta


def running_batchnorm_forward(x, gamma, beta, bn_param):
  """
  Forward pass for running batch normalization.
  
  Same as batch norm, but we normalize with running mean & variance instead of sample mean & variance
  What we expect is better robustness with non-i.i.d minibatches
  At init, momentum = 0

  Input:
  - x: Data of shape (N, D)
  - gamma: Scale parameter of shape (D,)
  - beta: Shift paremeter of shape (D,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features

  Returns a tuple of:
  - out: of shape (N, D)
  - cache: A tuple of values needed in the backward pass
  """
  mode = bn_param['mode']
  eps = bn_param.get('eps',0)
  momentum = bn_param.get('momentum', 0.1)
  
  if('ini' not in bn_param):
    momentum = 0
    bn_param['ini'] = False

  N, D = x.shape
  running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
  running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

  out, cache = None, None
  if mode == 'train':
    #############################################################################
    # TODO: Implement the training-time forward pass for batch normalization.   #
    # Use minibatch statistics to compute the mean and variance, use these      #
    # statistics to normalize the incoming data, and scale and shift the        #
    # normalized data using gamma and beta.                                     #
    #                                                                           #
    # You should store the output in the variable out. Any intermediates that   #
    # you need for the backward pass should be stored in the cache variable.    #
    #                                                                           #
    # You should also use your computed sample mean and variance together with  #
    # the momentum variable to update the running mean and running variance,    #
    # storing your result in the running_mean and running_var variables.        #
    #############################################################################
    mean = np.mean(x, 0)
    var = np.var(x,0)    
    
    running_mean = momentum * running_mean + (1 - momentum) * mean
    running_var = momentum * running_var + (1 - momentum) * var

    std = np.sqrt(running_var+eps)
    xn = (x-running_mean)/std
    out = xn * gamma + beta
    cache = (xn, mean, std, gamma, momentum)
        
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
  elif mode == 'test':
    #############################################################################
    # TODO: Implement the test-time forward pass for batch normalization. Use   #
    # the running mean and variance to normalize the incoming data, then scale  #
    # and shift the normalized data using gamma and beta. Store the result in   #
    # the out variable.                                                         #
    #############################################################################
    xn = (x-bn_param['running_mean'])/np.sqrt(bn_param['running_var']+eps)
    out = xn * gamma + beta
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
  else:
    raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

  # Store the updated running means back into bn_param
  bn_param['running_mean'] = running_mean
  bn_param['running_var'] = running_var

  return out, cache

def running_batchnorm_backward(dout, cache):
  """
  Backward pass for batch normalization.
  
  For this implementation, you should write out a computation graph for
  batch normalization on paper and propagate gradients backward through
  intermediate nodes.
  
  Inputs:
  - dout: Upstream derivatives, of shape (N, D)
  - cache: Variable of intermediates from batchnorm_forward.
  
  Returns a tuple of:
  - dx: Gradient with respect to inputs x, of shape (N, D)
  - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
  - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
  """
  dx, dgamma, dbeta = None, None, None
  #############################################################################
  # TODO: Implement the backward pass for batch normalization. Store the      #
  # results in the dx, dgamma, and dbeta variables.                           #
  #############################################################################
  xn, mean, std, gamma, momentum = cache
  m = xn.shape[0]
  dgamma = np.sum(xn*dout,axis=0)
  dbeta = np.sum(dout,axis=0)
  dxn = dout * gamma 
  dvar = -0.5 * np.sum(dxn*xn,0)/(std**2) * (1 - momentum)
  dmu = -np.sum(dxn,0)/std * (1 - momentum)                     
  dx = dxn/std + dmu/m + 2/m * dvar * xn * std
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return dx, dgamma, dbeta


def batchnorm_forward(x, gamma, beta, bn_param):
  """
  Forward pass for batch normalization.
  
  During training the sample mean and (uncorrected) sample variance are
  computed from minibatch statistics and used to normalize the incoming data.
  During training we also keep an exponentially decaying running mean of the mean
  and variance of each feature, and these averages are used to normalize data
  at test-time.

  At each timestep we update the running averages for mean and variance using
  an exponential decay based on the momentum parameter:

  running_mean = momentum * running_mean + (1 - momentum) * sample_mean
  running_var = momentum * running_var + (1 - momentum) * sample_var

  Note that the batch normalization paper suggests a different test-time
  behavior: they compute sample mean and variance for each feature using a
  large number of training images rather than using a running average. For
  this implementation we have chosen to use running averages instead since
  they do not require an additional estimation step; the torch7 implementation
  of batch normalization also uses running averages.

  Input:
  - x: Data of shape (N, D)
  - gamma: Scale parameter of shape (D,)
  - beta: Shift paremeter of shape (D,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features

  Returns a tuple of:
  - out: of shape (N, D)
  - cache: A tuple of values needed in the backward pass
  """
  mode = bn_param['mode']
  eps = bn_param.get('eps',0)
  momentum = bn_param.get('momentum', 0.9)
  noscale = bn_param.get('noscale',False)

  N, D = x.shape
  running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
  running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

  if(noscale):
    gamma, beta = 1.0, 0.0

  out, cache = None, None
  if mode == 'train':
    #############################################################################
    # TODO: Implement the training-time forward pass for batch normalization.   #
    # Use minibatch statistics to compute the mean and variance, use these      #
    # statistics to normalize the incoming data, and scale and shift the        #
    # normalized data using gamma and beta.                                     #
    #                                                                           #
    # You should store the output in the variable out. Any intermediates that   #
    # you need for the backward pass should be stored in the cache variable.    #
    #                                                                           #
    # You should also use your computed sample mean and variance together with  #
    # the momentum variable to update the running mean and running variance,    #
    # storing your result in the running_mean and running_var variables.        #
    #############################################################################
    mean = np.mean(x, 0)
    var = np.var(x,0)    
    std = np.sqrt(var+eps)
    xn = (x-mean)/std
    out = xn * gamma + beta
    cache = (xn, mean, std, gamma)
    running_mean = momentum * running_mean + (1 - momentum) * mean
    running_var = momentum * running_var + (1 - momentum) * var    
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
  elif mode == 'test':
    #############################################################################
    # TODO: Implement the test-time forward pass for batch normalization. Use   #
    # the running mean and variance to normalize the incoming data, then scale  #
    # and shift the normalized data using gamma and beta. Store the result in   #
    # the out variable.                                                         #
    #############################################################################
    xn = (x-bn_param['running_mean'])/np.sqrt(bn_param['running_var']+eps)
    out = xn * gamma + beta
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
  else:
    raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

  # Store the updated running means back into bn_param
  bn_param['running_mean'] = running_mean
  bn_param['running_var'] = running_var

  return out, cache


def batchnorm_backward(dout, cache):
  """
  Backward pass for batch normalization.
  
  For this implementation, you should write out a computation graph for
  batch normalization on paper and propagate gradients backward through
  intermediate nodes.
  
  Inputs:
  - dout: Upstream derivatives, of shape (N, D)
  - cache: Variable of intermediates from batchnorm_forward.
  
  Returns a tuple of:
  - dx: Gradient with respect to inputs x, of shape (N, D)
  - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
  - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
  """
  dx, dgamma, dbeta = None, None, None
  #############################################################################
  # TODO: Implement the backward pass for batch normalization. Store the      #
  # results in the dx, dgamma, and dbeta variables.                           #
  #############################################################################
  xn, mean, std, gamma = cache
  m = xn.shape[0]
  dgamma = np.sum(xn*dout,axis=0)
  dbeta = np.sum(dout,axis=0)
  dxn = dout * gamma 
  dvar = -0.5 * np.sum(dxn*xn,0)/(std**2) 
  dmu = -np.sum(dxn,0)/std                     
  dx = dxn/std + dmu/m + 2/m * dvar * xn * std
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
  """
  Alternative backward pass for batch normalization.
  
  For this implementation you should work out the derivatives for the batch
  normalizaton backward pass on paper and simplify as much as possible. You
  should be able to derive a simple expression for the backward pass.
  
  Note: This implementation should expect to receive the same cache variable
  as batchnorm_backward, but might not use all of the values in the cache.
  
  Inputs / outputs: Same as batchnorm_backward
  """
  dx, dgamma, dbeta = None, None, None
  #############################################################################
  # TODO: Implement the backward pass for batch normalization. Store the      #
  # results in the dx, dgamma, and dbeta variables.                           #
  #                                                                           #
  # After computing the gradient with respect to the centered inputs, you     #
  # should be able to compute gradients with respect to the inputs in a       #
  # single statement; our implementation fits on a single 80-character line.  #
  #############################################################################
  xn, mean, std, gamma = cache
  m = xn.shape[0]
  dgamma = np.sum(xn*dout,axis=0)
  dbeta = np.sum(dout,axis=0)
  dxn = dout * gamma 
  dx = (dxn - xn * np.mean(dxn*xn,0) - np.mean(dxn,0))/std     
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  
  return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
  """
  Performs the forward pass for (inverted) dropout.

  Inputs:
  - x: Input data, of any shape
  - dropout_param: A dictionary with the following keys:
    - p: Dropout parameter. We drop each neuron output with probability p.
    - mode: 'test' or 'train'. If the mode is train, then perform dropout;
      if the mode is test, then just return the input.
    - seed: Seed for the random number generator. Passing seed makes this
      function deterministic, which is needed for gradient checking but not in
      real networks.

  Outputs:
  - out: Array of the same shape as x.
  - cache: A tuple (dropout_param, mask). In training mode, mask is the dropout
    mask that was used to multiply the input; in test mode, mask is None.
  """
  p, mode = dropout_param['p'], dropout_param['mode']
  if 'seed' in dropout_param:
    np.random.seed(dropout_param['seed'])

  mask = None
  out = None

  if mode == 'train':
    ###########################################################################
    # TODO: Implement the training phase forward pass for inverted dropout.   #
    # Store the dropout mask in the mask variable.                            #
    ###########################################################################
    N, D = x.shape
    #mask = np.random.binomial([np.ones((N,D))],1-p)[0] * (1.0/(1-p))   
    mask = (np.random.rand(*x.shape) < (1.-p) ) / (1.-p)    #*x.shape -> return the vals & not the tuple, isn't inversed?
    out = x * mask
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
  elif mode == 'test':
    ###########################################################################
    # TODO: Implement the test phase forward pass for inverted dropout.       #
    ###########################################################################
    out = x # DOES NOT MULTIPLY BY 1/(1-p) SINCE IT IS ALREADY DONE IN TRAINING
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################

  cache = (dropout_param, mask)
  out = out.astype(x.dtype, copy=False)

  return out, cache


def dropout_backward(dout, cache):
  """
  Perform the backward pass for (inverted) dropout.

  Inputs:
  - dout: Upstream derivatives, of any shape
  - cache: (dropout_param, mask) from dropout_forward.
  """
  dropout_param, mask = cache
  mode = dropout_param['mode']
  
  dx = None
  if mode == 'train':
    ###########################################################################
    # TODO: Implement the training phase backward pass for inverted dropout.  #
    ###########################################################################
    dx = dout*mask
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
  elif mode == 'test':
    dx = dout
  return dx


def conv_forward_naive(x, w, b, conv_param):
  """
  A naive implementation of the forward pass for a convolutional layer.

  The input consists of N data points, each with C channels, height H and width
  W. We convolve each input with F different filters, where each filter spans
  all C channels and has height HH and width HH.

  Input:
  - x: Input data of shape (N, C, H, W)
  - w: Filter weights of shape (F, C, HH, WW)
  - b: Biases, of shape (F,)
  - conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.

  Returns a tuple of:
  - out: Output data, of shape (N, F, H', W') where H' and W' are given by
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
  - cache: (x, w, b, conv_param)
  """
  out = None
  #############################################################################
  # TODO: Implement the convolutional forward pass.                           #
  # Hint: you can use the function np.pad for padding.                        #
  #############################################################################
  s = conv_param['stride']
  pad = conv_param['pad']
  N, C, H, W = x.shape
  F, C, HH, WW = w.shape
  H2 = 1 + (H + 2 * pad - HH) / s
  W2 = 1 + (W + 2 * pad - WW) / s
  npad = ((0,0), (0,0), (pad,pad), (pad,pad))
  xpad = np.pad(x, pad_width=npad, mode='constant', constant_values=0)
  out = np.zeros((N, F, H2, W2))
  w2 = np.reshape(w,(F,C*HH*WW)).T

  for j in range(0,H,s):
    for i in range(0,W,s):
        window = np.reshape(xpad[:,:,j:j+HH,i:i+WW],(N,C*HH*WW))
        out[:,:,j/s,i/s] = window.dot(w2) + b # (N, D) * (D, F) => (N,F) @ (x,y)
  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, w, b, conv_param)
  return out, cache


def conv_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a convolutional layer.

  Inputs:
  - dout: Upstream derivatives.
  - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

  Returns a tuple of:
  - dx: Gradient with respect to x
  - dw: Gradient with respect to w
  - db: Gradient with respect to b
  """
  dx, dw, db = None, None, None
  #############################################################################
  # TODO: Implement the convolutional backward pass.                          #
  #############################################################################
  x, w, b, conv_param = cache
  s = conv_param['stride']
  pad = conv_param['pad']
  N, C, H, W = x.shape
  F, C, HH, WW = w.shape
  npad = ((0,0), (0,0), (pad,pad), (pad,pad))
  xpad = np.pad(x, pad_width=npad, mode='constant', constant_values=0)
  w2 = np.reshape(w,(F,C*HH*WW)).T
  #alloc dx, dw, db
  dx, dw, db = np.zeros_like(xpad), np.zeros_like(w), np.zeros_like(b)
  for j in range(0,H,s):
    for i in range(0,W,s):
        win = np.reshape(xpad[:,:,j:j+HH,i:i+WW],(N,C*HH*WW))
        dwin = dout[:,:,j/s,i/s].reshape((N,F))
        dx[:,:,j:j+HH,i:i+WW] += dwin.dot(w2.T).reshape((N,C,HH,WW)) #((N,F)*(F,D)), backprop convolution is convolution with filter.T
        dw += dwin.T.dot(win).reshape(w.shape)        
        db += np.sum(dwin,axis=0)
 
  dx = dx[:,:,pad:H+pad,pad:W+pad]
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx, dw, db


def max_pool_forward_naive(x, pool_param):
  """
  A naive implementation of the forward pass for a max pooling layer.

  Inputs:
  - x: Input data, of shape (N, C, H, W)
  - pool_param: dictionary with the following keys:
    - 'pool_height': The height of each pooling region
    - 'pool_width': The width of each pooling region
    - 'stride': The distance between adjacent pooling regions

  Returns a tuple of:
  - out: Output data
  - cache: (x, pool_param)
  """
  out = None
  #############################################################################
  # TODO: Implement the max pooling forward pass                              #
  #############################################################################
  WW = pool_param['pool_width']
  HH = pool_param['pool_height']
  s = pool_param['stride']
  N, C, H, W = x.shape
  out = np.zeros((N, C, H/s, W/s))
  mask = np.zeros((N, C, H/s, W/s),dtype=np.uint8)#uint8
  
  mg = np.mgrid[:N,:C]
  for j in range(0,H,s):
    for i in range(0,W,s):
       win = x[:,:,j:j+HH,i:i+WW].reshape((N,C,HH*WW))
       idx = win.argmax(axis=2)
       mask[...,j/s,i/s] = idx
       out[...,j/s,i/s] = win[mg[0],mg[1],idx]
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, mask, pool_param)
  return out, cache


def max_pool_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a max pooling layer.

  Inputs:
  - dout: Upstream derivatives
  - cache: A tuple of (x, pool_param) as in the forward pass.

  Returns:
  - dx: Gradient with respect to x
  """
  dx = None
  #############################################################################
  # TODO: Implement the max pooling backward pass                             #
  #############################################################################
  x, mask, pool_param = cache
  WW = pool_param['pool_width']
  HH = pool_param['pool_height']
  s = pool_param['stride']
  N, C, H, W = x.shape  
  dx = np.zeros_like(x)
  mg = np.mgrid[:N,:C]
  for j in range(0,H,s):
    for i in range(0,W,s): 
      dwin = dout[:,:,j/s,i/s]
      mwin = mask[:,:,j/s,i/s]
      xx = (mwin%WW).astype(int) 
      yy = (mwin/WW).astype(int)
      dx[mg[0],mg[1],j+yy,i+xx] = dwin
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
  """
  Computes the forward pass for spatial batch normalization.
  
  Inputs:
  - x: Input data of shape (N, C, H, W)
  - gamma: Scale parameter, of shape (C,)
  - beta: Shift parameter, of shape (C,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance. momentum=0 means that
      old information is discarded completely at every time step, while
      momentum=1 means that new information is never incorporated. The
      default of momentum=0.9 should work well in most situations.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features
    
  Returns a tuple of:
  - out: Output data, of shape (N, C, H, W)
  - cache: Values needed for the backward pass
  """
  out, cache = None, None
  #############################################################################
  # TODO: Implement the forward pass for spatial batch normalization.         #
  #                                                                           #
  # HINT: You can implement spatial batch normalization using the vanilla     #
  # version of batch normalization defined above. Your implementation should  #
  # be very short; ours is less than five lines.                              #
  #############################################################################
  N, C, H, W = x.shape
  xr = np.rollaxis(x,1,4).reshape((N*H*W,C))
  xn, cache = batchnorm_forward(xr,gamma,beta,bn_param) #cache holds (xn, mean, std, gamma)
  out = np.rollaxis( xn.reshape((N,H,W,C)),3,1)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return out, cache


def spatial_batchnorm_backward(dout, cache):
  """
  Computes the backward pass for spatial batch normalization.
  
  Inputs:
  - dout: Upstream derivatives, of shape (N, C, H, W)
  - cache: Values from the forward pass
  
  Returns a tuple of:
  - dx: Gradient with respect to inputs, of shape (N, C, H, W)
  - dgamma: Gradient with respect to scale parameter, of shape (C,)
  - dbeta: Gradient with respect to shift parameter, of shape (C,)
  """
  dx, dgamma, dbeta = None, None, None

  #############################################################################
  # TODO: Implement the backward pass for spatial batch normalization.        #
  #                                                                           #
  # HINT: You can implement spatial batch normalization using the vanilla     #
  # version of batch normalization defined above. Your implementation should  #
  # be very short; ours is less than five lines.                              #
  #############################################################################
  N, C, H, W = dout.shape
  doutr = np.rollaxis(dout,1,4).reshape((N*H*W,C))
  dan, dgamma, dbeta = batchnorm_backward(doutr, cache)
  dx = np.rollaxis( dan.reshape((N,H,W,C)),3,1)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return dx, dgamma, dbeta
  

def svm_loss(x, y):
  """
  Computes the loss and gradient using for multiclass SVM classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  N = x.shape[0]
  correct_class_scores = x[np.arange(N), y]
  margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
  margins[np.arange(N), y] = 0
  loss = np.sum(margins) / N
  num_pos = np.sum(margins > 0, axis=1)
  dx = np.zeros_like(x)
  dx[margins > 0] = 1
  dx[np.arange(N), y] -= num_pos
  dx /= N
  return loss, dx


def softmax_loss(x, y):
  """
  Computes the loss and gradient for softmax classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  probs = np.exp(x - np.max(x, axis=1, keepdims=True))
  probs /= np.sum(probs, axis=1, keepdims=True)
  N = x.shape[0]
  loss = -np.sum(np.log(probs[np.arange(N), y])) / N #faster than prob -1
  dx = probs.copy()
  dx[np.arange(N), y] -= 1
  dx /= N
  return loss, dx

def euclidean_loss(x, y):
    """
  Computes the loss and gradient for least-square regression.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
    #loss is 1/2 * mean( (y-x)²)
    num = y.shape[0]
    diff = (y-x)
    loss = np.sum(diff**2) / num / 2.

    dx = -1 * diff / num
    return loss, dx

