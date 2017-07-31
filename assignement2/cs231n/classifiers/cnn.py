import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *
from cs231n.classifiers.fc_net import FullyConnectedNet


class ThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - affine - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32, weight_norm=False):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype

    self.use_batchnorm = True
    self.use_weightnorm = False
    wn_var = 0.05
    
    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################

    C, H, W = input_dim
    s = 1
    pad = (filter_size - 1) / 2  
    H2 = 1 + (H + 2 * pad - filter_size) / s
    W2 = 1 + (W + 2 * pad - filter_size) / s   
    pool1_dim = num_filters * H2 * W2 / 4

    self.params['W1'] = weight_scale * np.random.randn(num_filters, C, filter_size, filter_size)
    self.params['b1'] = np.zeros(num_filters)    

    if(self.use_batchnorm):
        self.params['gamma1'] = weight_scale * np.random.randn(num_filters)
        self.params['beta1'] = weight_scale * np.random.randn(num_filters)
    #elif(self.use_weightnorm):
    #    self.params['v1'] = wn_var * np.random.randn(num_filters, C, filter_size, filter_size)
    #    self.params['g1'] = 1.0 * np.random.randn(num_filters)

    self.params['W2'] = weight_scale * np.random.randn(pool1_dim, hidden_dim)
    self.params['b2'] = np.zeros(hidden_dim)

    if(self.use_batchnorm):
        self.params['gamma2'] = weight_scale * np.random.randn(hidden_dim)
        self.params['beta2'] = weight_scale * np.random.randn(hidden_dim)
    #elif(self.use_weightnorm):
    #    self.params['v2'] = wn_var * np.random.randn(pool1_dim, hidden_dim)
    #    self.params['g2'] = 1.0 * np.random.randn(hidden_dim)


    self.params['W3'] = weight_scale * np.random.randn(hidden_dim, num_classes)
    self.params['b3'] = np.zeros(num_classes)


    if(self.use_weightnorm):
        self.params['v3'] = wn_var * np.random.randn(hidden_dim, num_classes)
        self.params['g3'] = 1.0 * np.random.randn(num_classes)
    
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.items():
      self.params[k] = v.astype(dtype)

    self.bn_params = []
    self.bn_params = [{'mode': 'train'} for i in range(2)]  
     
 
  def loss(self, X, y=None):
    mode = 'test' if y is None else 'train'
    for bn_param in self.bn_params:
        bn_param['mode'] = mode
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': int((filter_size - 1) / 2)}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################  
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']

    '''
    if(self.use_weightnorm):
        g1, v1 = self.params['g1'], self.params['v1']
        W1, cache1 = weightnorm_forward(g1,v1)
        cache_wn1 = cache1
        
        g2, v2 = self.params['g2'], self.params['v2']
        W2, cache2 = weightnorm_forward(g2,v2)
        cache_wn2 = cache2

        g3, v3 = self.params['g3'], self.params['v3']
        W3, cache3 = weightnorm_forward(g3,v3)
        cache_wn3 = cache3
    '''

    if(self.use_batchnorm):
        gamma1, beta1 = self.params['gamma1'], self.params['beta1']
        gamma2, beta2 = self.params['gamma2'], self.params['beta2']
        y1, cache1 = conv_bn_relu_pool_forward(X, W1, b1, gamma1, beta1, conv_param, pool_param, self.bn_params[0])       
        y2, cache2 = affine_relu_bn_forward(y1, W2, b2, gamma2, beta2, self.bn_params[1])   
    else:
        y1, cache1 = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)     
        y2, cache2 = affine_relu_forward(y1,W2,b2)

    
    scores, cache3 = affine_forward(y2,W3,b3)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    loss, dloss = softmax_loss(scores,y)
    dy3, dW3, db3 = affine_backward(dloss,cache3)

    if(self.use_batchnorm):
        dy2, dW2, db2, dgamma2, dbeta2  = affine_relu_bn_backward(dy3, cache2)
        dy1, dW1, db1, dgamma1, dbeta1 = conv_bn_relu_pool_backward(dy2,cache1)
        grads['gamma1'] = dgamma1 
        grads['beta1'] = dbeta1
        grads['gamma2'] = dgamma2 
        grads['beta2'] = dbeta2
    else:
        dy2, dW2, db2 = affine_relu_backward(dy3, cache2)
        dy1, dW1, db1 = conv_relu_pool_backward(dy2,cache1)

    #print(self.reg)
    '''
    if(self.use_weightnorm):
        dg3, dv3 = weightnorm_backward(dW3, cache_wn3)
        dg2, dv2 = weightnorm_backward(dW2, cache_wn2)
        dg1, dv1 = weightnorm_backward(dW1, cache_wn1)
        
        #dv3 += self.reg * dv3 
        #dv2 += self.reg * dv2 
        #dv1 += self.reg * dv1

        grads['v3'], grads['g3'] = dv3, g3
        grads['v2'], grads['g2'] = dv2, g2
        grads['v1'], grads['g1'] = dv1, g1
    '''


    #regularization
    smooth = self.reg * (np.sum(np.sum(W1**2)) + np.sum(np.sum(W2**2)) + np.sum(np.sum(W3**2)))/2
    loss += smooth
    dW3 += self.reg * W3 
    dW2 += self.reg * W2 
    dW1 += self.reg * W1 

    grads['W3'] = dW3
    grads['b3'] = db3
    grads['W2'] = dW2
    grads['b2'] = db2
    grads['W1'] = dW1
    grads['b1'] = db1





    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads
  
  
pass
