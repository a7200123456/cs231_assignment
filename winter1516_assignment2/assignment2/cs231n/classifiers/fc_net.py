import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *

def affine_bn_relu_forward(x, w, b, gamma, beta, bn_param):
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
  bn_a, bn_cache = batchnorm_forward(a, gamma, beta, bn_param)
  out, relu_cache = relu_forward(bn_a)
  cache = (fc_cache, bn_cache, relu_cache)
  return out, cache


def affine_bn_relu_backward(dout, cache):
  """
  Backward pass for the affine-relu convenience layer
  """
  fc_cache, bn_cache, relu_cache = cache
  dbn_a = relu_backward(dout, relu_cache)
  da, dgamma, dbeta = batchnorm_backward(dbn_a, bn_cache)
  dx, dw, db = affine_backward(da, fc_cache)
  return dx, dw, db, dgamma, dbeta


pass


class TwoLayerNet(object):
  """
  A two-layer fully-connected neural network with ReLU nonlinearity and
  softmax loss that uses a modular layer design. We assume an input dimension
  of D, a hidden dimension of H, and perform classification over C classes.
  
  The architecure should be affine - relu - affine - softmax.

  Note that this class does not implement gradient descent; instead, it
  will interact with a separate Solver object that is responsible for running
  optimization.

  The learnable parameters of the model are stored in the dictionary
  self.params that maps parameter names to numpy arrays.
  """
  
  def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
               weight_scale=1e-3, reg=0.0):
    """
    Initialize a new network.

    Inputs:
    - input_dim: An integer giving the size of the input
    - hidden_dim: An integer giving the size of the hidden layer
    - num_classes: An integer giving the number of classes to classify
    - dropout: Scalar between 0 and 1 giving dropout strength.
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - reg: Scalar giving L2 regularization strength.
    """
    self.params = {}
    self.reg = reg
    
    ############################################################################
    # TODO: Initialize the weights and biases of the two-layer net. Weights    #
    # should be initialized from a Gaussian with standard deviation equal to   #
    # weight_scale, and biases should be initialized to zero. All weights and  #
    # biases should be stored in the dictionary self.params, with first layer  #
    # weights and biases using the keys 'W1' and 'b1' and second layer weights #
    # and biases using the keys 'W2' and 'b2'.                                 #
    ############################################################################
    
    self.params['W1'] = weight_scale * np.random.randn(input_dim, hidden_dim)
    self.params['b1'] = np.zeros(hidden_dim)
    self.params['W2'] = weight_scale * np.random.randn(hidden_dim, num_classes)
    self.params['b2'] = np.zeros(num_classes)
    self.Dropout_ratio = 0
    #pass
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################


  def loss(self, X, y=None):
    """
    Compute loss and gradient for a minibatch of data.

    Inputs:
    - X: Array of input data of shape (N, d_1, ..., d_k)
    - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

    Returns:
    If y is None, then run a test-time forward pass of the model and return:
    - scores: Array of shape (N, C) giving classification scores, where
      scores[i, c] is the classification score for X[i] and class c.

    If y is not None, then run a training-time forward and backward pass and
    return a tuple of:
    - loss: Scalar value giving the loss
    - grads: Dictionary with the same keys as self.params, mapping parameter
      names to gradients of the loss with respect to those parameters.
    """  
    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the two-layer net, computing the    #
    # class scores for X and storing them in the scores variable.              #
    ############################################################################
    # Unpack variables from the params dictionary
    W1, b1 = self.params['W1']/(1-self.Dropout_ratio), self.params['b1']
    W2, b2 = self.params['W2']/(1-self.Dropout_ratio), self.params['b2']
    X_2D = np.reshape(X,(X.shape[0],np.prod(X.shape[1:])))
    N, D = X_2D.shape

    import theano
    import theano.tensor as T
    import numpy
    from itertools import izip

    X_m = T.matrix()
    W1_m = theano.shared(W1)
    b1_v = theano.shared(b1)
    W2_m = theano.shared(W2)
    b2_v = theano.shared(b2)

    z1 = T.dot(X_m,W1_m) + T.transpose((b1_v).dimshuffle(0,'x'))
    a1 = T.maximum(z1,0)
    z2 = T.dot(a1,W2_m) + T.transpose((b2_v).dimshuffle(0,'x'))

    score_fnt = theano.function(
                  inputs = [X_m],
                  outputs = z2
                  #,on_unused_input='ignore'
                  )

    scores = score_fnt(X_2D)

    pass
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # If y is None then we are in test mode so just return scores
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the two-layer net. Store the loss  #
    # in the loss variable and gradients in the grads dictionary. Compute data #
    # loss using softmax, and make sure that grads[k] holds the gradients for  #
    # self.params[k]. Don't forget to add L2 regularization!                   #
    #                                                                          #
    # NOTE: To ensure that your implementation matches ours and you pass the   #
    # automated tests, make sure that your L2 regularization includes a factor #
    # of 0.5 to simplify the expression for the gradient.                      #
    ############################################################################import theano
    import theano.tensor as T
    import numpy
    from itertools import izip

    y_m = T.matrix()
    X_m = T.matrix()
    W1_m = theano.shared(W1)
    b1_v = theano.shared(b1)
    W2_m = theano.shared(W2)
    b2_v = theano.shared(b2)

    z1 = T.dot(X_m,W1_m) + T.transpose((b1_v).dimshuffle(0,'x'))
    a1 = T.maximum(z1,0)
    z2 = T.dot(a1,W2_m) + T.transpose((b2_v).dimshuffle(0,'x'))
    exp_z2 = T.exp(z2)
    sum_z2 = T.sum(exp_z2,axis = 1)
    loss_m = -y_m*T.log(exp_z2/(sum_z2.dimshuffle(0,'x')))
    loss_s = T.sum(loss_m)/N + 0.5*self.reg*(T.sum(W1_m**2)+T.sum(W2_m**2))

    dW1,db1,dW2,db2 = T.grad(loss_s,[W1_m,b1_v,W2_m,b2_v])

    loss_fnt = theano.function(
                  inputs = [X_m,y_m],
                  outputs = [loss_s,dW1,db1,dW2,db2]
                  #,on_unused_input='ignore'
                  )


    y_matrix = np.zeros((N,W2.shape[1]))
    for n in range(N):
      y_matrix[n,y[n]] = 1
    
    loss , dW1,db1,dW2,db2 = loss_fnt(X_2D,y_matrix)

    grads['W1'] = dW1
    grads['b1'] = db1
    grads['W2'] = dW2
    grads['b2'] = db2
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads


class FullyConnectedNet(object):
  """
  A fully-connected neural network with an arbitrary number of hidden layers,
  ReLU nonlinearities, and a softmax loss function. This will also implement
  dropout and batch normalization as options. For a network with L layers,
  the architecture will be
  
  {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax
  
  where batch normalization and dropout are optional, and the {...} block is
  repeated L - 1 times.
  
  Similar to the TwoLayerNet above, learnable parameters are stored in the
  self.params dictionary and will be learned using the Solver class.
  """

  def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
               dropout=0, use_batchnorm=False, reg=0.0,
               weight_scale=1e-2, dtype=np.float32, seed=None):
    """
    Initialize a new FullyConnectedNet.
    
    Inputs:
    - hidden_dims: A list of integers giving the size of each hidden layer.
    - input_dim: An integer giving the size of the input.
    - num_classes: An integer giving the number of classes to classify.
    - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
      the network should not use dropout at all.
    - use_batchnorm: Whether or not the network should use batch normalization.
    - reg: Scalar giving L2 regularization strength.
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - dtype: A numpy datatype object; all computations will be performed using
      this datatype. float32 is faster but less accurate, so you should use
      float64 for numeric gradient checking.
    - seed: If not None, then pass this random seed to the dropout layers. This
      will make the dropout layers deteriminstic so we can gradient check the
      model.
    """
    self.use_batchnorm = use_batchnorm
    self.use_dropout = dropout > 0
    self.reg = reg
    self.num_layers = 1 + len(hidden_dims)
    self.dtype = dtype
    self.params = {}

    ############################################################################
    # TODO: Initialize the parameters of the network, storing all values in    #
    # the self.params dictionary. Store weights and biases for the first layer #
    # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
    # initialized from a normal distribution with standard deviation equal to  #
    # weight_scale and biases should be initialized to zero.                   #
    #                                                                          #
    # When using batch normalization, store scale and shift parameters for the #
    # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
    # beta2, etc. Scale parameters should be initialized to one and shift      #
    # parameters should be initialized to zero.                                #
    ############################################################################
    for i in range(self.num_layers):
      if i == 0:
        self.params['W'+str(i+1)] = weight_scale * np.random.randn(input_dim, hidden_dims[i])
        self.params['b'+str(i+1)] = np.zeros(hidden_dims[i])
      elif i == (self.num_layers-1):
        self.params['W'+str(i+1)] = weight_scale * np.random.randn(hidden_dims[i-1], num_classes)
        self.params['b'+str(i+1)] = np.zeros(num_classes)
      else:
        self.params['W'+str(i+1)] = weight_scale * np.random.randn(hidden_dims[i-1], hidden_dims[i])
        self.params['b'+str(i+1)] = np.zeros(hidden_dims[i])

      if self.use_batchnorm:
        if i < (self.num_layers-1):
          self.params['gamma'+str(i+1)] = np.ones(hidden_dims[i])
          self.params['beta'+str(i+1)] = np.zeros(hidden_dims[i])
    pass
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # When using dropout we need to pass a dropout_param dictionary to each
    # dropout layer so that the layer knows the dropout probability and the mode
    # (train / test). You can pass the same dropout_param to each dropout layer.
    self.dropout_param = {}
    if self.use_dropout:
      self.dropout_param = {'mode': 'train', 'p': dropout}
      if seed is not None:
        self.dropout_param['seed'] = seed
    
    # With batch normalization we need to keep track of running means and
    # variances, so we need to pass a special bn_param object to each batch
    # normalization layer. You should pass self.bn_params[0] to the forward pass
    # of the first batch normalization layer, self.bn_params[1] to the forward
    # pass of the second batch normalization layer, etc.
    self.bn_params = []
    if self.use_batchnorm:
      self.bn_params = [{'mode': 'train'} for i in xrange(self.num_layers - 1)]
    
    # Cast all parameters to the correct datatype
    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)


  def loss(self, X, y=None):
    """
    Compute loss and gradient for the fully-connected net.

    Input / output: Same as TwoLayerNet above.
    """
    X = X.astype(self.dtype)
    mode = 'test' if y is None else 'train'

    # Set train/test mode for batchnorm params and dropout param since they
    # behave differently during training and testing.
    if self.dropout_param is not None:
      self.dropout_param['mode'] = mode   
    if self.use_batchnorm:
      for bn_param in self.bn_params:
        bn_param[mode] = mode

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the fully-connected net, computing  #
    # the class scores for X and storing them in the scores variable.          #
    #                                                                          #
    # When using dropout, you'll need to pass self.dropout_param to each       #
    # dropout forward pass.                                                    #
    #                                                                          #
    # When using batch normalization, you'll need to pass self.bn_params[0] to #
    # the forward pass for the first batch normalization layer, pass           #
    # self.bn_params[1] to the forward pass for the second batch normalization #
    # layer, etc.                                                              #
    ############################################################################
    out = {}
    cache = {}
    dropout_cache = {}
    for i in range(self.num_layers):
      if i == 0:
        if self.use_batchnorm:
          out['L'+str(i+1)], cache['L'+str(i+1)] = affine_bn_relu_forward(X, self.params['W'+str(i+1)], \
            self.params['b'+str(i+1)],self.params['gamma'+str(i+1)], self.params['beta'+str(i+1)], self.bn_params[i])
        else:
          out['L'+str(i+1)], cache['L'+str(i+1)] = affine_relu_forward(X, self.params['W'+str(i+1)], self.params['b'+str(i+1)])
        #self.params['z'+str(i+1)] = np.dot(X,self.params['W'+str(i+1)])+ (self.params['b'+str(i+1)]).T
        #self.params['a'+str(i+1)] = np.maximum(self.params['z'+str(i+1)],0)
      elif i == self.num_layers-1:
        if self.use_batchnorm:
          out['L'+str(i+1)], cache['L'+str(i+1)] = affine_forward(out['L'+str(i)], self.params['W'+str(i+1)], self.params['b'+str(i+1)])
        else:
          out['L'+str(i+1)], cache['L'+str(i+1)] = affine_relu_forward(out['L'+str(i)], self.params['W'+str(i+1)], self.params['b'+str(i+1)])
      else:
        if self.use_batchnorm:
          out['L'+str(i+1)], cache['L'+str(i+1)] = affine_bn_relu_forward(out['L'+str(i)], self.params['W'+str(i+1)], \
            self.params['b'+str(i+1)],self.params['gamma'+str(i+1)], self.params['beta'+str(i+1)], self.bn_params[i])
        else:
          out['L'+str(i+1)], cache['L'+str(i+1)] = affine_relu_forward(out['L'+str(i)], self.params['W'+str(i+1)], self.params['b'+str(i+1)])
        #self.params['z'+str(i+1)] = np.dot(self.params['a'+str(i)],self.params['W'+str(i+1)])+ (self.params['b'+str(i+1)]).T
        #self.params['a'+str(i+1)] = np.maximum(self.params['z'+str(i+1)],0)

      if self.use_dropout: 
        out['L'+str(i+1)], dropout_cache['L'+str(i+1)] = dropout_forward(out['L'+str(i+1)], self.dropout_param)

    scores = out['L'+str(self.num_layers)]
    #print scores
    pass
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # If test mode return early
    if mode == 'test':
      return scores

    loss, grads = 0.0, {}
    ############################################################################
    # TODO: Implement the backward pass for the fully-connected net. Store the #
    # loss in the loss variable and gradients in the grads dictionary. Compute #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    #                                                                          #
    # When using batch normalization, you don't need to regularize the scale   #
    # and shift parameters.                                                    #
    #                                                                          #
    # NOTE: To ensure that your implementation matches ours and you pass the   #
    # automated tests, make sure that your L2 regularization includes a factor #
    # of 0.5 to simplify the expression for the gradient.                      #
    ############################################################################
    N = scores.shape[0]
    num_classes = scores.shape[1]
    
    # softmax loss
    y_matrix = np.zeros(scores.shape)
    for n in range(N):
      y_matrix[n,y[n]] = 1

    exp_scores = np.exp(scores- np.max(scores, axis=1, keepdims=True))
    sum_exp_scores = np.sum(exp_scores,axis = 1)
    loss_m = -y_matrix*np.log(exp_scores/(sum_exp_scores[:,None]+1e-5))

    W_total = 0;
    for i in range(self.num_layers):
        W_total += np.sum(self.params['W'+str(i+1)]**2)
    
    loss_s = np.sum(loss_m)/N + 0.5*self.reg*W_total
    loss = loss_s
    grads_X = (exp_scores/(sum_exp_scores[:,None])) - y_matrix
    
    # backprop 
    for i in range(self.num_layers):
      cur_layer = self.num_layers -i

      if self.use_dropout:
        grads_X = dropout_backward(grads_X, dropout_cache['L'+str(cur_layer)])

      if self.use_batchnorm:
        if cur_layer == self.num_layers:
          grads_X, grads['W'+str(cur_layer)], grads['b'+str(cur_layer)] = \
                  affine_backward(grads_X, cache['L'+str(cur_layer)])  
        else:
          grads_X, grads['W'+str(cur_layer)], grads['b'+str(cur_layer)], grads['gamma'+str(cur_layer)], grads['beta'+str(cur_layer)] = \
          affine_bn_relu_backward(grads_X, cache['L'+str(cur_layer)])
      else:
        grads_X, grads['W'+str(cur_layer)], grads['b'+str(cur_layer)] = \
                  affine_relu_backward(grads_X, cache['L'+str(cur_layer)])  

      grads['W'+str(cur_layer)] /= N
      grads['W'+str(cur_layer)] += self.reg*self.params['W'+str(cur_layer)]
      grads['b'+str(cur_layer)] /= N
      if self.use_batchnorm:
        if cur_layer < self.num_layers:
          grads['gamma'+str(cur_layer)] /= N
          grads['beta'+str(cur_layer)] /= N

    pass
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    #print grads['W1'].shape
    return loss, grads
