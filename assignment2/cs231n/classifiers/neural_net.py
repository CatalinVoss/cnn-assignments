import numpy as np
import matplotlib.pyplot as plt

# Helpers
relu = lambda x: x * (x > 0)

def init_two_layer_model(input_size, hidden_size, output_size):
  """
  Initialize the weights and biases for a two-layer fully connected neural
  network. The net has an input dimension of D, a hidden layer dimension of H,
  and performs classification over C classes. Weights are initialized to small
  random values and biases are initialized to zero.

  Inputs:
  - input_size: The dimension D of the input data
  - hidden_size: The number of neurons H in the hidden layer
  - ouput_size: The number of classes C

  Returns:
  A dictionary mapping parameter names to arrays of parameter values. It has
  the following keys:
  - W1: First layer weights; has shape (D, H)
  - b1: First layer biases; has shape (H,)
  - W2: Second layer weights; has shape (H, C)
  - b2: Second layer biases; has shape (C,)
  """
  # initialize a model
  model = {}
  model['W1'] = 0.00001 * np.random.randn(input_size, hidden_size)
  model['b1'] = np.zeros(hidden_size)
  model['W2'] = 0.00001 * np.random.randn(hidden_size, output_size)
  model['b2'] = np.zeros(output_size)
  return model

def two_layer_net(X, model, y=None, reg=0.0):
  """
  Compute the loss and gradients for a two layer fully connected neural network.
  The net has an input dimension of D, a hidden layer dimension of H, and
  performs classification over C classes. We use a softmax loss function and L2
  regularization the the weight matrices. The two layer net should use a ReLU
  nonlinearity after the first affine layer.

  The two layer net has the following architecture:

  input - fully connected layer - ReLU - fully connected layer - softmax

  The outputs of the second fully-connected layer are the scores for each
  class.

  Inputs:
  - X: Input data of shape (N, D). Each X[i] is a training sample.
  - model: Dictionary mapping parameter names to arrays of parameter values.
    It should contain the following:
    - W1: First layer weights; has shape (D, H)
    - b1: First layer biases; has shape (H,)
    - W2: Second layer weights; has shape (H, C)
    - b2: Second layer biases; has shape (C,)
  - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
    an integer in the range 0 <= y[i] < C. This parameter is optional; if it
    is not passed then we only return scores, and if it is passed then we
    instead return the loss and gradients.
  - reg: Regularization strength.

  Returns:
  If y not is passed, return a matrix scores of shape (N, C) where scores[i, c]
  is the score for class c on input X[i].

  If y is not passed, instead return a tuple of:
  - loss: Loss (data loss and regularization loss) for this batch of training
    samples.
  - grads: Dictionary mapping parameter names to gradients of those parameters
    with respect to the loss function. This should have the same keys as model.
  """

  # unpack variables from the model dictionary
  W1,b1,W2,b2 = model['W1'], model['b1'], model['W2'], model['b2']
  N, D = X.shape

  # compute the forward pass
  scores = None
  #############################################################################
  # Perform the forward pass, computing the class scores for the input.       #
  # Store the result in the scores variable, which should be an array of      #
  # shape (N, C).                                                             #
  #############################################################################
  layer1 = X.dot(W1)+b1
  layer2 = relu(layer1)
  layer3 = layer2.dot(W2) + b2

  scores = layer3 # no need to make two variables, but more readable this way. #relu(X.dot(W1)+b1).dot(W2) + b2
    
  # If the targets are not given then jump out, we're done
  if y is None:
    return scores

  #############################################################################
  # Compute Softmax classifier loss with L2 regularization                    #
  # log and e cancel out, sum splits                                          #
  # softmax loss is -sum(correct scores) + log(sum(e^[all scores (for each    #
  #  respective example)]))                                                   #
  #############################################################################
  rows = np.sum(np.exp(layer3), axis=1)
  layer4 = np.sum(-layer3[range(N), y]+np.log(rows) ) / N
  # loss = np.sum(-scores[range(N), y]+np.log(np.sum(np.exp(scores), axis=1)) ) / N
  # loss += 0.5 * reg * (np.sum(W1*W1) + np.sum(W2*W2))
  loss = layer4 + 0.5 * reg * (np.sum(W1*W1) + np.sum(W2*W2))

  # compute the gradients
  grads = {}

  #############################################################################
  # Compute the backward pass, computing the derivatives of the weights       #
  # and biases. Store the results in the grads dictionary. For example,       #
  # grads['W1'] should store the gradient on W1, and be a matrix of same size #
  #############################################################################

  dlayer4 = 1.0

  # Gradient of np.log(np.sum(np.exp(layer3), axis=1))
  dlayer3 = (np.exp(layer3).T / rows).T
  # Gradient of -layer3[range(N), y]:
  ys = np.zeros(dlayer3.shape)
  ys[range(N), y] = 1
  dlayer3 -= ys
  # / N term
  dlayer3 /= N
  # Chain rule
  dlayer3 *= dlayer4

  # Chain rule, element-wise multiplication works out
  dlayer2 = dlayer3.dot(W2.T)

  # Relu gradient
  dlayer1 = dlayer2 * (layer1>=0)

  dW1 = X.T.dot(dlayer1)
  dW2 = layer2.T.dot(dlayer3)

  # Same as matrix multiplication with 1-vector, chain rule works out
  db1 = np.sum(dlayer1, axis=0)
  db2 = np.sum(dlayer3, axis=0)

  # Regularization
  dW1 += reg * W1
  dW2 += reg * W2

  # Store
  grads['W1'] = dW1
  grads['W2'] = dW2
  grads['b1'] = db1
  grads['b2'] = db2

  return loss, grads

