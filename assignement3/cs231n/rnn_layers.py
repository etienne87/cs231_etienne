import numpy as np


"""
This file defines layer types that are commonly used for recurrent neural
networks.
"""

nonlinearity = 'relu'


def rnn_step_forward(x, prev_h, Wx, Wh, b):
  """
  Run the forward pass for a single timestep of a vanilla RNN that uses a tanh
  activation function.

  The input data has dimension D, the hidden state has dimension H, and we use
  a minibatch size of N.

  Inputs:
  - x: Input data for this timestep, of shape (N, D).
  - prev_h: Hidden state from previous timestep, of shape (N, H)
  - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
  - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
  - b: Biases of shape (H,)

  Returns a tuple of:
  - next_h: Next hidden state, of shape (N, H)
  - cache: Tuple of values needed for the backward pass.
  """
  next_h, cache = None, None
  ##############################################################################
  # TODO: Implement a single forward step for the vanilla RNN. Store the next  #
  # hidden state and any values you need for the backward pass in the next_h   #
  # and cache variables respectively.                                          #
  ##############################################################################
  xw = x.dot(Wx)
  hw = prev_h.dot(Wh)
  
  next_h_lin = xw+hw+b
  if nonlinearity == 'tanh':
    next_h = np.tanh(next_h_lin)
  else:
    next_h = np.maximum(next_h_lin, 0)
  cache = x, prev_h, Wx, Wh, b, next_h 
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  return next_h, cache


def rnn_step_backward(dnext_h, cache):
  """
  Backward pass for a single timestep of a vanilla RNN.
  
  Inputs:
  - dnext_h: Gradient of loss with respect to next hidden state
  - cache: Cache object from the forward pass
  
  Returns a tuple of:
  - dx: Gradients of input data, of shape (N, D)
  - dprev_h: Gradients of previous hidden state, of shape (N, H)
  - dWx: Gradients of input-to-hidden weights, of shape (N, H)
  - dWh: Gradients of hidden-to-hidden weights, of shape (H, H)
  - db: Gradients of bias vector, of shape (H,)
  """
  dx, dprev_h, dWx, dWh, db = None, None, None, None, None
  ##############################################################################
  # TODO: Implement the backward pass for a single step of a vanilla RNN.      #
  #                                                                            #
  # HINT: For the tanh function, you can compute the local derivative in terms #
  # of the output value from tanh.                                             #
  ##############################################################################
  x, prev_h, Wx, Wh, b, next_h = cache
  if nonlinearity == 'tanh':
    dy = dnext_h*(1-next_h*next_h)
  else:
    dy = np.where(next_h > 0, dnext_h, 0)

  dx = dy.dot(Wx.T)
  dprev_h = dy.dot(Wh.T)
  dWx = x.T.dot(dy)
  dWh = prev_h.T.dot(dy)
  db = np.sum(dy,axis=0)
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  return dx, dprev_h, dWx, dWh, db


def rnn_forward(x, h0, Wx, Wh, b):
  """
  Run a vanilla RNN forward on an entire sequence of data. We assume an input
  sequence composed of T vectors, each of dimension D. The RNN uses a hidden
  size of H, and we work over a minibatch containing N sequences. After running
  the RNN forward, we return the hidden states for all timesteps.
  
  Inputs:
  - x: Input data for the entire timeseries, of shape (N, T, D).
  - h0: Initial hidden state, of shape (N, H)
  - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
  - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
  - b: Biases of shape (H,)
  
  Returns a tuple of:
  - h: Hidden states for the entire timeseries, of shape (N, T, H).
  - cache: Values needed in the backward pass
  """
  h, cache = None, None
  ##############################################################################
  # TODO: Implement forward pass for a vanilla RNN running on a sequence of    #
  # input data. You should use the rnn_step_forward function that you defined  #
  # above.                                                                     #
  ##############################################################################
  N, T, D = x.shape
  H = Wx.shape[1]
  h = np.zeros((N,T,H))
  cache = []
  #first step : no prev_h
  h[:,0,:], _ = rnn_step_forward(x[:,0,:],h0, Wx, Wh, b)
  for i in range(1,T):
    xi = x[:,i,:]
    prev_h = h[:,i-1,:]
    h[:,i,:], _ = rnn_step_forward(x[:,i,:], prev_h, Wx, Wh, b)

  cache = x, h0, Wx, Wh, b, h
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  return h, cache


def rnn_backward(dh, cache):
  """
  Compute the backward pass for a vanilla RNN over an entire sequence of data.
  
  Inputs:
  - dh: Upstream gradients of all hidden states, of shape (N, T, H)
  
  Returns a tuple of:
  - dx: Gradient of inputs, of shape (N, T, D)
  - dh0: Gradient of initial hidden state, of shape (N, H)
  - dWx: Gradient of input-to-hidden weights, of shape (D, H)
  - dWh: Gradient of hidden-to-hidden weights, of shape (H, H)
  - db: Gradient of biases, of shape (H,)
  """
  dx, dh0, dWx, dWh, db = None, None, None, None, None
  ##############################################################################
  # TODO: Implement the backward pass for a vanilla RNN running an entire      #
  # sequence of data. You should use the rnn_step_backward function that you   #
  # defined above.                                                             #
  ##############################################################################
  x, h0, Wx, Wh, b, h = cache
  dx = np.zeros_like(x)
  dh0 = np.zeros_like(h0)
  dWx = np.zeros_like(Wx)
  dWh = np.zeros_like(Wh)
  db = np.zeros_like(b)
  dhcur = dh.copy()
  N, T, D = x.shape
  for i in reversed(range(1,T)):
    dnext_h = dhcur[:,i,:]
    next_h = h[:,i,:]
    prev_h = h[:,i-1,:]
    cache_loc = x[:,i,:], prev_h, Wx, Wh, b, next_h
    dx_, dprev_h, dWx_, dWh_, db_ = rnn_step_backward(dnext_h, cache_loc)
    dhcur[:,i-1,:] += dprev_h
    dx[:,i,:] = dx_
    dWx += dWx_
    dWh += dWh_
    db += db_
  #final backprop
  dnext_h = dhcur[:,0,:]
  next_h = h[:,0,:]
  prev_h = h0
  cache_loc = x[:,0,:], prev_h, Wx, Wh, b, next_h
  dx_, dh0, dWx_, dWh_, db_ = rnn_step_backward(dnext_h, cache_loc)
  dx[:,0,:] = dx_
  dWx += dWx_
  dWh += dWh_
  db += db_
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  return dx, dh0, dWx, dWh, db


def word_embedding_forward(x, W):
  """
  Forward pass for word embeddings. We operate on minibatches of size N where
  each sequence has length T. We assume a vocabulary of V words, assigning each
  to a vector of dimension D.
  
  Inputs:
  - x: Integer array of shape (N, T) giving indices of words. Each element idx
    of x muxt be in the range 0 <= idx < V.
  - W: Weight matrix of shape (V, D) giving word vectors for all words.
  
  Returns a tuple of:
  - out: Array of shape (N, T, D) giving word vectors for all input words.
  - cache: Values needed for the backward pass
  """
  out, cache = None, None
  ##############################################################################
  # TODO: Implement the forward pass for word embeddings.                      #
  #                                                                            #
  # HINT: This should be very simple.                                          #
  ##############################################################################
  N, T = x.shape
  V = W.shape[0]
  x2 = np.zeros((N*T,V))
  x2[np.arange(N*T),x.ravel()] = 1
  x2 = x2.reshape((N,T,V))
  out = x2.dot(W)
  cache = x2, W
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  return out, cache


def word_embedding_backward(dout, cache):
  """
  Backward pass for word embeddings. We cannot back-propagate into the words
  since they are integers, so we only return gradient for the word embedding
  matrix.
  
  HINT: Look up the function np.add.at
  
  Inputs:
  - dout: Upstream gradients of shape (N, T, D)
  - cache: Values from the forward pass
  
  Returns:
  - dW: Gradient of word embedding matrix, of shape (V, D).
  """
  dW = None
  ##############################################################################
  # TODO: Implement the backward pass for word embeddings.                     #
  #                                                                            #
  # HINT: Look up the function np.add.at                                       #
  ##############################################################################
  x, W = cache
  dW = np.zeros_like(W)
  for t in range(x.shape[1]):
    dW += x[:,t,:].T.dot(dout[:,t,:]) # (V,N) * (N,D)
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  return dW


def sigmoid(x):
  """
  A numerically stable version of the logistic sigmoid function.
  """
  pos_mask = (x >= 0)
  neg_mask = (x < 0)
  z = np.zeros_like(x)
  z[pos_mask] = np.exp(-x[pos_mask])
  z[neg_mask] = np.exp(x[neg_mask])
  top = np.ones_like(x)
  top[neg_mask] = z[neg_mask]
  return top / (1 + z)

def sigmoid_backward(dout, sigma):
    return dout*sigma*(1-sigma)

def tanh_backward(dout, tanh):
    return dout*(1-tanh*tanh)

def lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b):
  """
  Forward pass for a single timestep of an LSTM.
  
  The input data has dimension D, the hidden state has dimension H, and we use
  a minibatch size of N.
  
  Inputs:
  - x: Input data, of shape (N, D)
  - prev_h: Previous hidden state, of shape (N, H)
  - prev_c: previous cell state, of shape (N, H)
  - Wx: Input-to-hidden weights, of shape (D, 4H)
  - Wh: Hidden-to-hidden weights, of shape (H, 4H)
  - b: Biases, of shape (4H,)
  
  Returns a tuple of:
  - next_h: Next hidden state, of shape (N, H)
  - next_c: Next cell state, of shape (N, H)
  - cache: Tuple of values needed for backward pass.
  """
  next_h, next_c, cache = None, None, None
  #############################################################################
  # TODO: Implement the forward pass for a single timestep of an LSTM.        #
  # You may want to use the numerically stable sigmoid implementation above.  #
  #############################################################################
  #gates
  N, D = x.shape
  N, H = prev_h.shape
  a = x.dot(Wx) + prev_h.dot(Wh) + b
  i,f,o,g = sigmoid(a[:,:H]), sigmoid(a[:,H:2*H]), sigmoid(a[:,2*H:3*H]), np.tanh(a[:,3*H:4*H])
  next_c = f*prev_c + i*g
  next_h = o*np.tanh(next_c)
  cache = i, f, o, g, x, prev_c, prev_h, next_c, next_h, Wx, Wh, b
  
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  
  return next_h, next_c, cache


def lstm_step_backward(dnext_h, dnext_c, cache):
  """
  Backward pass for a single timestep of an LSTM.
  
  Inputs:
  - dnext_h: Gradients of next hidden state, of shape (N, H)
  - dnext_c: Gradients of next cell state, of shape (N, H)
  - cache: Values from the forward pass
  
  Returns a tuple of:
  - dx: Gradient of input data, of shape (N, D)
  - dprev_h: Gradient of previous hidden state, of shape (N, H)
  - dprev_c: Gradient of previous cell state, of shape (N, H)
  - dWx: Gradient of input-to-hidden weights, of shape (D, 4H)
  - dWh: Gradient of hidden-to-hidden weights, of shape (H, 4H)
  - db: Gradient of biases, of shape (4H,)
  """
  dx, dh, dc, dWx, dWh, db = None, None, None, None, None, None
  #############################################################################
  # TODO: Implement the backward pass for a single timestep of an LSTM.       #
  #                                                                           #
  # HINT: For sigmoid and tanh you can compute local derivatives in terms of  #
  # the output value from the nonlinearity.                                   #
  #############################################################################
  i, f, o, g, x, prev_c, prev_h, next_c, next_h, Wx, Wh, b = cache
  yc = np.tanh(next_c)
  do = dnext_h * yc
  dnext_c += dnext_h * o * (1-yc*yc)  #dc is used many times...
  df = dnext_c * prev_c 
  di = dnext_c * g
  dg = dnext_c * i
  dprev_c = dnext_c * f
  da = di * (1-i) * i
  da = np.append(da,df * (1-f)*f, axis=1) #dfs
  da = np.append(da,do * (1-o)*o, axis=1) #dos
  da = np.append(da,dg * (1-g*g), axis=1) #dgs
  dprev_h = da.dot(Wh.T).reshape(prev_h.shape)
  dx = da.dot(Wx.T).reshape(x.shape)
  dWx = x.reshape(x.shape[0], -1).T.dot(da)
  dWh = prev_h.reshape(prev_h.shape[0], -1).T.dot(da)
  db = np.sum(da, axis=0)

  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################

  return dx, dprev_h, dprev_c, dWx, dWh, db


def lstm_forward(x, h0, Wx, Wh, b):
  """
  Forward pass for an LSTM over an entire sequence of data. We assume an input
  sequence composed of T vectors, each of dimension D. The LSTM uses a hidden
  size of H, and we work over a minibatch containing N sequences. After running
  the LSTM forward, we return the hidden states for all timesteps.
  
  Note that the initial cell state is passed as input, but the initial cell
  state is set to zero. Also note that the cell state is not returned; it is
  an internal variable to the LSTM and is not accessed from outside.
  
  Inputs:
  - x: Input data of shape (N, T, D)
  - h0: Initial hidden state of shape (N, H)
  - Wx: Weights for input-to-hidden connections, of shape (D, 4H)
  - Wh: Weights for hidden-to-hidden connections, of shape (H, 4H)
  - b: Biases of shape (4H,)
  
  Returns a tuple of:
  - h: Hidden states for all timesteps of all sequences, of shape (N, T, H)
  - cache: Values needed for the backward pass.
  """
  h, cache = None, None
  #############################################################################
  # TODO: Implement the forward pass for an LSTM over an entire timeseries.   #
  # You should use the lstm_step_forward function that you just defined.      #
  #############################################################################
  N, T, D = x.shape
  H = Wh.shape[0]
  h = np.zeros((N,T,H))
  c = np.zeros((N,T,H))
  i = np.zeros((N,T,H))
  f = np.zeros((N,T,H))
  o = np.zeros((N,T,H))
  g = np.zeros((N,T,H))
  c0 = np.zeros((N,H))
  #first step : no prev_h
  h[:,0,:], c[:,0,:], cache_loc = lstm_step_forward(x[:,0,:], h0, c0, Wx, Wh, b)
  i[:,0,:], f[:,0,:], o[:,0,:], g[:,0,:] = cache_loc[:4]
  for t in range(1,T):
    prev_h = h[:,t-1,:]
    prev_c = c[:,t-1,:]
    h[:,t,:], c[:,t,:], cache_loc = lstm_step_forward(x[:,t,:], prev_h, prev_c, Wx, Wh, b)
    i[:,t,:], f[:,t,:], o[:,t,:], g[:,t,:] = cache_loc[:4]

  cache = i,f,o,g,x,h0,h,c,Wx,Wh,b   
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################

  return h, cache


def lstm_backward(dh, cache):
  """
  Backward pass for an LSTM over an entire sequence of data.]
  
  Inputs:
  - dh: Upstream gradients of hidden states, of shape (N, T, H)
  - cache: Values from the forward pass
  
  Returns a tuple of:
  - dx: Gradient of input data of shape (N, T, D)
  - dh0: Gradient of initial hidden state of shape (N, H)
  - dWx: Gradient of input-to-hidden weight matrix of shape (D, 4H)
  - dWh: Gradient of hidden-to-hidden weight matrix of shape (H, 4H)
  - db: Gradient of biases, of shape (4H,)
  """
  dx, dh0, dWx, dWh, db = None, None, None, None, None
  #############################################################################
  # TODO: Implement the backward pass for an LSTM over an entire timeseries.  #
  # You should use the lstm_step_backward function that you just defined.     #
  #############################################################################
  
  i,f,o,g,x,h0,h,c,Wx,Wh,b = cache
  dx = np.zeros_like(x)
  dh0 = np.zeros_like(h0)
  dWx = np.zeros_like(Wx)
  dWh = np.zeros_like(Wh)
  db = np.zeros_like(b)
  dc = np.zeros_like(c)
  dh = dh.copy() #important for gradient checking
  N, T, D = x.shape
  for t in reversed(range(1,T)):
    dnext_h = dh[:,t,:]
    dnext_c = dc[:,t,:]
    next_h = h[:,t,:]
    prev_h = h[:,t-1,:]
    next_c = c[:,t,:]
    prev_c = c[:,t-1,:]
    it,ft,ot,gt = i[:,t,:],f[:,t,:],o[:,t,:],g[:,t,:]    
    cache_loc = it, ft, ot, gt, x[:,t,:], prev_c, prev_h, next_c, next_h, Wx, Wh, b 
    dx_, dprev_h, dprev_c, dWx_, dWh_, db_ = lstm_step_backward(dnext_h, dnext_c, cache_loc)
    dh[:,t-1,:] += dprev_h
    dc[:,t-1,:] += dprev_c
    dx[:,t,:] = dx_
    dWx += dWx_
    dWh += dWh_
    db += db_
  #final backprop
  dnext_h = dh[:,0,:]
  dnext_c = dc[:,0,:]
  next_h = h[:,0,:]
  prev_h = h0
  next_c = c[:,0,:]
  prev_c = np.zeros_like(next_c)
  it,ft,ot,gt = i[:,0,:],f[:,0,:],o[:,0,:],g[:,0,:] 
  cache_loc = it, ft, ot, gt, x[:,0,:], prev_c, prev_h, next_c, next_h, Wx, Wh, b 
  dx_, dh0, dc0, dWx_, dWh_, db_ = lstm_step_backward(dnext_h, dnext_c, cache_loc)
  dx[:,0,:] = dx_
  dWx += dWx_
  dWh += dWh_
  db += db_

  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  
  return dx, dh0, dWx, dWh, db


def temporal_affine_forward(x, w, b):
  """
  Forward pass for a temporal affine layer. The input is a set of D-dimensional
  vectors arranged into a minibatch of N timeseries, each of length T. We use
  an affine function to transform each of those vectors into a new vector of
  dimension M.

  Inputs:
  - x: Input data of shape (N, T, D)
  - w: Weights of shape (D, M)
  - b: Biases of shape (M,)
  
  Returns a tuple of:
  - out: Output data of shape (N, T, M)
  - cache: Values needed for the backward pass
  """
  N, T, D = x.shape
  M = b.shape[0]
  out = x.reshape(N * T, D).dot(w).reshape(N, T, M) + b
  cache = x, w, b, out
  return out, cache


def temporal_affine_backward(dout, cache):
  """
  Backward pass for temporal affine layer.

  Input:
  - dout: Upstream gradients of shape (N, T, M)
  - cache: Values from forward pass

  Returns a tuple of:
  - dx: Gradient of input, of shape (N, T, D)
  - dw: Gradient of weights, of shape (D, M)
  - db: Gradient of biases, of shape (M,)
  """
  x, w, b, out = cache
  N, T, D = x.shape
  M = b.shape[0]

  dx = dout.reshape(N * T, M).dot(w.T).reshape(N, T, D)
  dw = dout.reshape(N * T, M).T.dot(x.reshape(N * T, D)).T
  db = dout.sum(axis=(0, 1))

  return dx, dw, db


def temporal_softmax_loss(x, y, mask, verbose=False):
  """
  A temporal version of softmax loss for use in RNNs. We assume that we are
  making predictions over a vocabulary of size V for each timestep of a
  timeseries of length T, over a minibatch of size N. The input x gives scores
  for all vocabulary elements at all timesteps, and y gives the indices of the
  ground-truth element at each timestep. We use a cross-entropy loss at each
  timestep, summing the loss over all timesteps and averaging across the
  minibatch.

  As an additional complication, we may want to ignore the model output at some
  timesteps, since sequences of different length may have been combined into a
  minibatch and padded with NULL tokens. The optional mask argument tells us
  which elements should contribute to the loss.

  Inputs:
  - x: Input scores, of shape (N, T, V)
  - y: Ground-truth indices, of shape (N, T) where each element is in the range
       0 <= y[i, t] < V
  - mask: Boolean array of shape (N, T) where mask[i, t] tells whether or not
    the scores at x[i, t] should contribute to the loss.

  Returns a tuple of:
  - loss: Scalar giving loss
  - dx: Gradient of loss with respect to scores x.
  """

  N, T, V = x.shape
  
  x_flat = x.reshape(N * T, V)
  y_flat = y.reshape(N * T)
  mask_flat = mask.reshape(N * T)
  
  probs = np.exp(x_flat - np.max(x_flat, axis=1, keepdims=True))
  probs /= np.sum(probs, axis=1, keepdims=True)
  loss = -np.sum(mask_flat * np.log(probs[np.arange(N * T), y_flat])) / N
  dx_flat = probs.copy()
  dx_flat[np.arange(N * T), y_flat] -= 1
  dx_flat /= N
  dx_flat *= mask_flat[:, None]
  
  if verbose: print('dx_flat: ', dx_flat.shape)
  
  dx = dx_flat.reshape(N, T, V)
  
  return loss, dx

