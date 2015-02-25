import numpy as np

class LinearClassifier:

  def __init__(self):
    self.W = None

  def train(self, X, y, learning_rate=1e-3, reg=1e-5, num_iters=100,
            batch_size=200, verbose=False):
    """
    Train this linear classifier using stochastic gradient descent.

    Inputs:
    - X: D x N array of training data. Each training point is a D-dimensional
         column.
    - y: 1-dimensional array of length N with labels 0...K-1, for K classes.
    - learning_rate: (float) learning rate for optimization.
    - reg: (float) regularization strength.
    - num_iters: (integer) number of steps to take when optimizing
    - batch_size: (integer) number of training examples to use at each step.
    - verbose: (boolean) If true, print progress during optimization.

    Outputs:
    A list containing the value of the loss function at each training iteration.
    """
    dim, num_train = X.shape
    num_classes = np.max(y) + 1 # assume y takes values 0...K-1 where K is number of classes
    if self.W is None:
      # lazily initialize W
      self.W = np.random.randn(num_classes, dim) * 0.001

    # Run stochastic gradient descent to optimize W
    loss_history = []
    for it in xrange(num_iters):
      X_batch = None
      y_batch = None

      batch_mask = np.random.choice(num_train, batch_size)
      X_batch = X[:,batch_mask]
      y_batch = y[batch_mask]

      # evaluate loss and gradient
      loss, grad = self.loss(X_batch, y_batch, reg)
      loss_history.append(loss)

      # perform parameter update
      step = -learning_rate * grad
      self.W += step

      if verbose and it % 100 == 0:
        print 'iteration %d / %d: loss %f' % (it, num_iters, loss)

    return loss_history

  def predict(self, X):
    """
    Use the trained weights of this linear classifier to predict labels for
    data points.

    Inputs:
    - X: D x N array of training data. Each column is a D-dimensional point.

    Returns:
    - y_pred: Predicted labels for the data in X. y_pred is a 1-dimensional
      array of length N, and each element is an integer giving the predicted
      class.
    """
    y_pred = np.zeros(X.shape[1])
    scores = self.W.dot(X)
    y_pred = np.argmax(scores, axis=0) # top scoring class
    return y_pred
  
  def loss(self, X_batch, y_batch, reg):
    """
    Compute the loss function and its derivative. 
    Subclasses will override this.

    Inputs:
    - X_batch: D x N array of data; each column is a data point.
    - y_batch: 1-dimensional array of length N with labels 0...K-1, for K classes.
    - reg: (float) regularization strength.

    Returns: A tuple containing:
    - loss as a single float
    - gradient with respect to self.W; an array of the same shape as W
    """
    pass


class LinearSVM(LinearClassifier):
  """ A subclass that uses the Multiclass SVM loss function """

  def loss(self, X_batch, y_batch, reg):
    return svm_loss_vectorized(self.W, X_batch, y_batch, reg)


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  D, num_train = X.shape
  scores = W.dot(X)
  correct_class_scores = scores[y, range(num_train)]
  margins = np.maximum(0, scores - correct_class_scores + 1.0)
  margins[y, range(num_train)] = 0

  loss_cost = np.sum(margins) / num_train
  loss_reg = 0.5 * reg * np.sum(W * W)
  loss = loss_cost + loss_reg
  num_pos = np.sum(margins > 0, axis=0) # number of positive losses

  dscores = np.zeros(scores.shape)
  dscores[margins > 0] = 1
  dscores[y, range(num_train)] = -num_pos

  dW = dscores.dot(X.T) / num_train + reg * W

  return loss, dW


class Softmax(LinearClassifier):
  """ A subclass that uses the Softmax + Cross-entropy loss function """

  def loss(self, X_batch, y_batch, reg):
    return softmax_loss_vectorized(self.W, X_batch, y_batch, reg)


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  D, num_train = X.shape
  scores = W.dot(X) # C x N

  scores -= np.max(scores, axis = 0)
  p = np.exp(scores)
  p /= np.sum(p, axis = 0)

  loss_cost = -np.sum(np.log(p[y, range(y.size)])) / num_train
  loss_reg = 0.5 * reg * np.sum(W * W)
  loss = loss_cost + loss_reg

  dscores = p
  dscores[y, range(y.size)] -= 1.0
  dW = dscores.dot(X.T) / num_train + reg * W

  return loss, dW
