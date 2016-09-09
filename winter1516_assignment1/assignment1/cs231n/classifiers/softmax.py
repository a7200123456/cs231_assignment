import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_classes = W.shape[1]
  num_train = X.shape[0]
  y_matrix = np.zeros((num_train,num_classes))
  for n in range(num_train):
    y_matrix[n,y[n]] = 1

  import theano
  import theano.tensor as T

  y_m = T.matrix()
  X_m = T.matrix()
  W_m = T.matrix()
  XW_m = T.dot(X_m,W_m) 
  expXW_m = T.exp(XW_m)
  sum_expXW_v = T.sum(expXW_m,axis = 1)
  loss_m = -y_m*T.log(expXW_m/(sum_expXW_v.dimshuffle(0,'x')))
  loss_s = T.sum(loss_m)/num_train
  dW = T.grad(loss_s,W_m)
  #for n in range(num_train):
  #  T.set_subtensor(loss_s[n], -T.log(expXW_s[n,y_s[n]]/sum_expXW_s[n]))

  softmax_layer = theano.function(
                  inputs = [X_m,W_m,y_m],
                  outputs = [loss_s,dW]
                  #,on_unused_input='ignore'
                  )

  loss,dW = softmax_layer(X,W,y_matrix)
  #pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_classes = W.shape[1]
  num_train = X.shape[0]
  y_matrix = np.zeros((num_train,num_classes))
  for n in range(num_train):
    y_matrix[n,y[n]] = 1
    
  import theano
  import theano.tensor as T

  reg_s = T.scalar()
  y_m = T.matrix()
  X_m = T.matrix()
  W_m = T.matrix()
  XW_m = T.dot(X_m,W_m) 
  expXW_m = T.exp(XW_m)
  sum_expXW_v = T.sum(expXW_m,axis = 1)
  loss_m = -y_m*T.log(expXW_m/(sum_expXW_v.dimshuffle(0,'x')))
  loss_s = T.sum(loss_m)/num_train + reg_s*T.sum(W_m**2)
  dW = T.grad(loss_s,W_m)
  #for n in range(num_train):
  #  T.set_subtensor(loss_s[n], -T.log(expXW_s[n,y_s[n]]/sum_expXW_s[n]))

  softmax_layer = theano.function(
                  inputs = [X_m,W_m,y_m,reg_s],
                  outputs = [loss_s,dW]
                  #,on_unused_input='ignore'
                  )

  loss,dW = softmax_layer(X,W,y_matrix,reg)
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

