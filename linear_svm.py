from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

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
    dW = np.zeros(W.shape) # initialize the gradient as zero

    # compute the loss and the gradient

    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            #去除j=y[i]的部分
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1 # note delta = 1
            #取max(0,sum(scores-correct_score))
            if margin > 0:
                loss += margin

                #此处对于加入的两行代码的位置存疑
            #？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？


#为什么dW的运算在if margin>0之下呢？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？


                # 在此加入计算梯度的部分
                #  loss funtion=y_predict - y[i]，对第j行W的求导所得即为X第i行
                # W在y_predict的计算式中是被转置的，所以这里是第j列，
                # 而此处的X与W的行列数不同，
                # 所以转置
                dW[:, j] += X[i, :].T   # .T代表矩阵的转置

                dW[:, y[i]] -= X[i, :].T

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    #dW也要取平均
    dW /= num_train
    # Add regularization to the loss.
    #正则化的方式是用常数乘W^2
    loss += reg * np.sum(W * W)
    # dW也需要正则化！！！
    dW += 2*reg*W
    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    #要让计算loss和计算derivative同时进行,修改上面计算Loss的部分代码，补充
    pass
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    return loss, dW



def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape) # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]
    # score + correct_class_score + delta = margin
    score = X.dot(W)

    #不明觉厉
    #为什么一定要np.arange而不能用slice呢？？？？？？？？？？
    # numpy.arange(start, stop, step, dtype = None)
    correct_class_score = score[np.arange(num_train),y]
    #关键：把correct_class_score拉成一个长向量
    correct_class_score = np.reshape(correct_class_score,(num_train, -1))
    #

    margin = score - correct_class_score + 1 #set delta=1
    margin[margin < 0] = 0 #去掉等于y的部分
    #get max(0,margin)
    margin[np.arange(num_train),y] = 0
    loss = np.sum(margin)/num_train
    loss += reg * np.sum(W * W)
    pass


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    margin[margin > 0] = 1  # 每个符合条件的输出在该位置贡献xi

    row_sum = np.sum(margin, axis=1)  # 每一个样本中符合条件的输出数

    margin[np.arange(num_train), y] -= row_sum;  # 每个符合条件的输出在标记处贡献-xi

    dW = np.dot(X.T, margin)
    dW = dW/num_train
    dW += 2*reg*W;
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
