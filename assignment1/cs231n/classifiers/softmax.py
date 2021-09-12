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
    X_len = X.shape[0]
    num_classes = W.shape[1]
    
    score = X.dot(W)
    for i in range(X_len):
        score_sum = np.sum(np.exp(score[i,:])) #분모
        loss += -score[i,y[i]] + np.log(score_sum)
        dW[:,y[i]] += -1*X[i,:]
        for j in range(num_classes):
            dW[:,j] += X[i,:]*np.exp(score[i,j])/score_sum
    
    loss = loss/X_len 
    loss = loss + reg*np.sum(W*W)
    dW = dW/X_len
    dW = dW + 2*reg*W
    
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
    X_len = X.shape[0]
    num_classes = W.shape[1]
    
    score = X.dot(W) #NxC
    score_exp = np.exp(score)
    sum_exp_scores = np.sum(score_exp,axis=1) #N개의 분모들
    
    loss = -np.sum(score[list(range(X_len)),y]) + np.sum(np.log(sum_exp_scores))
    loss = loss/X_len + reg*np.sum(W*W)
    
    exp_term = score_exp/(sum_exp_scores.reshape(-1,1)) #NxC / Nx1
    exp_term[list(range(X_len)),y] -= 1 #true label 부분
    dW = X.T.dot(exp_term)
    dW = dW/X_len + 2*reg*W
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW

