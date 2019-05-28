import numpy as np
from random import shuffle
from past.builtins import xrange

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
  num_train = X.shape[0]
  for i in range(num_train):
      temp = X[i].dot(W)
      f = np.max(temp)
      temp -= f
      total_score = np.sum(np.exp(temp))
      loss -= np.log(np.exp(temp[y[i]]) / np.sum(np.exp(temp)))
      for j in range(W.shape[1]):
          dW[:,j] += X[i].T * (np.exp(temp[j]) / total_score)
          if j == y[i]:
              dW[:, y[i]] += - X[i].T
              
  loss /= num_train
  loss += reg * np.sum(W * W)
  
  dW /= num_train
  dW += 2 * reg * W
  
  
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
  num_train = X.shape[0]
  scores = X.dot(W)
  scores -= np.max(scores)
  total_score = np.sum(np.exp(scores), axis = 1)
  loss_matrix = np.exp(scores[np.arange(scores.shape[0]), y]) / total_score
  loss_matrix = - np.log(loss_matrix)
  loss = np.sum(loss_matrix) / num_train
  loss += reg * np.sum(W * W)
  
  coef_matrix = np.exp(scores)
  coef_matrix = coef_matrix.T / total_score.T
  coef_matrix = coef_matrix.T
  coef_matrix[np.arange(coef_matrix.shape[0]), y] -= 1
  dW = (X.T).dot(coef_matrix)
  dW /= num_train
  dW += 2 * reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

