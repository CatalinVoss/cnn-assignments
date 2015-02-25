import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


def two_layer_convnet(X, model, y=None, reg=0.0, dropout=1.0):
  """
  Compute the loss and gradient for a simple two-layer ConvNet. The architecture
  is conv-relu-pool-affine-softmax, where the conv layer uses stride-1 "same"
  convolutions to preserve the input size; the pool layer uses non-overlapping
  2x2 pooling regions. We use L2 regularization on both the convolutional layer
  weights and the affine layer weights.

  Inputs:
  - X: Input data, of shape (N, C, H, W)
  - model: Dictionary mapping parameter names to parameters. A two-layer Convnet
    expects the model to have the following parameters:
    - W1, b1: Weights and biases for the convolutional layer
    - W2, b2: Weights and biases for the affine layer
  - y: Vector of labels of shape (N,). y[i] gives the label for the point X[i].
  - reg: Regularization strength.

  Returns:
  If y is None, then returns:
  - scores: Matrix of scores, where scores[i, c] is the classification score for
    the ith input and class c.

  If y is not None, then returns a tuple of:
  - loss: Scalar value giving the loss.
  - grads: Dictionary with the same keys as model, mapping parameter names to
    their gradients.
  """
  
  # Unpack weights
  W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
  N, C, H, W = X.shape

  # We assume that the convolution is "same", so that the data has the same
  # height and width after performing the convolution. We can then use the
  # size of the filter to figure out the padding.
  conv_filter_height, conv_filter_width = W1.shape[2:]
  assert conv_filter_height == conv_filter_width, 'Conv filter must be square'
  assert conv_filter_height % 2 == 1, 'Conv filter height must be odd'
  assert conv_filter_width % 2 == 1, 'Conv filter width must be odd'
  conv_param = {'stride': 1, 'pad': (conv_filter_height - 1) / 2}
  pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
  dropout_param = {'p': dropout}
  dropout_param['mode'] = 'test' if y is None else 'train'

  # Compute the forward pass
  a1, cache1 = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
  d1, cache2 = dropout_forward(a1, dropout_param)
  scores, cache3 = affine_forward(d1, W2, b2)

  if y is None:
    return scores

  # Compute the backward pass
  data_loss, dscores = softmax_loss(scores, y)

  # Compute the gradients using a backward pass
  dd1, dW2, db2 = affine_backward(dscores, cache3)
  da1 = dropout_backward(dd1, cache2)
  dX,  dW1, db1 = conv_relu_pool_backward(da1, cache1)

  # Add regularization
  dW1 += reg * W1
  dW2 += reg * W2
  reg_loss = 0.5 * reg * sum(np.sum(W * W) for W in [W1, W2])

  loss = data_loss + reg_loss
  grads = {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2}
  
  return loss, grads


def init_two_layer_convnet(weight_scale=1e-3, bias_scale=0, input_shape=(3, 32, 32),
                           num_classes=10, num_filters=32, filter_size=5):
  """
  Initialize the weights for a two-layer ConvNet.

  Inputs:
  - weight_scale: Scale at which weights are initialized. Default 1e-3.
  - bias_scale: Scale at which biases are initialized. Default is 0.
  - input_shape: Tuple giving the input shape to the network; default is
    (3, 32, 32) for CIFAR-10.
  - num_classes: The number of classes for this network. Default is 10
    (for CIFAR-10)
  - num_filters: The number of filters to use in the convolutional layer.
  - filter_size: The width and height for convolutional filters. We assume that
    all convolutions are "same", so we pick padding to ensure that data has the
    same height and width after convolution. This means that the filter size
    must be odd.

  Returns:
  A dictionary mapping parameter names to numpy arrays containing:
    - W1, b1: Weights and biases for the convolutional layer
    - W2, b2: Weights and biases for the fully-connected layer.
  """
  C, H, W = input_shape
  assert filter_size % 2 == 1, 'Filter size must be odd; got %d' % filter_size

  model = {}
  model['W1'] = weight_scale * np.random.randn(num_filters, C, filter_size, filter_size)
  model['b1'] = bias_scale * np.random.randn(num_filters)
  model['W2'] = weight_scale * np.random.randn(num_filters * H * W / 4, num_classes)
  model['b2'] = bias_scale * np.random.randn(num_classes)
  return model


def init_three_layer_convnet(input_shape=(3, 32, 32), num_classes=10,
                            filter_size=5, num_filters=(32, 128),
                            weight_scale=1e-2, bias_scale=0, dtype=np.float32):
  """
  Initialize a three layer ConvNet with the following architecture:

  conv - relu - pool - affine - relu - dropout - affine - softmax

  The convolutional layer uses stride 1 and has padding to perform "same"
  convolution, and the pooling layer is 2x2 stride 2.

  Inputs:
  - input_shape: Tuple (C, H, W) giving the shape of each training sample.
    Default is (3, 32, 32) for CIFAR-10.
  - num_classes: Number of classes over which classification will be performed.
    Default is 10 for CIFAR-10.
  - filter_size: The height and width of filters in the convolutional layer.
  - num_filters: Tuple (F, H) where F is the number of filters to use in the
    convolutional layer and H is the number of neurons to use in the hidden
    affine layer.
  - weight_scale: Weights are initialized from a gaussian distribution with
    standard deviation equal to weight_scale.
  - bias_scale: Biases are initialized from a gaussian distribution with
    standard deviation equal to bias_scale.
  - dtype: Numpy datatype used to store parameters. Default is float32 for
    speed.
  """
  C, H, W = input_shape
  F1, FC = num_filters
  filter_size = 5
  model = {}
  model['W1'] = np.random.randn(F1, 3, filter_size, filter_size)
  model['b1'] = np.random.randn(F1)
  model['W2'] = np.random.randn(H * W * F1 / 4, FC)
  model['b2'] = np.random.randn(FC)
  model['W3'] = np.random.randn(FC, num_classes)
  model['b3'] = np.random.randn(num_classes)

  for i in [1, 2, 3]:
    model['W%d' % i] *= weight_scale
    model['b%d' % i] *= bias_scale

  for k in model:
    model[k] = model[k].astype(dtype, copy=False)

  return model


def three_layer_convnet(X, model, y=None, reg=0.0, dropout=None):
  """
  Compute the loss and gradient for a simple three layer ConvNet that uses
  the following architecture:

  conv - relu - pool - affine - relu - dropout - affine - softmax

  The convolution layer uses stride 1 and sets the padding to achieve "same"
  convolutions, and the pooling layer is 2x2 stride 2. We use L2 regularization
  on all weights, and no regularization on the biases.

  Inputs:
  - X: (N, C, H, W) array of input data
  - model: Dictionary mapping parameter names to values; it should contain
    the following parameters:
    - W1, b1: Weights and biases for convolutional layer
    - W2, b2, W3, b3: Weights and biases for affine layers
  - y: Integer array of shape (N,) giving the labels for the training samples
    in X. This is optional; if it is not given then return classification
    scores; if it is given then instead return loss and gradients.
  - reg: The regularization strength.
  - dropout: The dropout parameter. If this is None then we skip the dropout
    layer; this allows this function to work even before the dropout layer
    has been implemented.
  """
  W1, b1 = model['W1'], model['b1']
  W2, b2 = model['W2'], model['b2']
  W3, b3 = model['W3'], model['b3']

  conv_param = {'stride': 1, 'pad': (W1.shape[2] - 1) / 2}
  pool_param = {'stride': 2, 'pool_height': 2, 'pool_width': 2}
  dropout_param = {'p': dropout}
  dropout_param['mode'] = 'test' if y is None else 'train'

  a1, cache1 = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
  a2, cache2 = affine_relu_forward(a1, W2, b2)
  if dropout is None:
    scores, cache4 = affine_forward(a2, W3, b3)
  else:
    d2, cache3 = dropout_forward(a2, dropout_param)
    scores, cache4 = affine_forward(d2, W3, b3)

  if y is None:
    return scores

  data_loss, dscores = softmax_loss(scores, y)
  if dropout is None:
    da2, dW3, db3 = affine_backward(dscores, cache4)
  else:
    dd2, dW3, db3 = affine_backward(dscores, cache4)
    da2 = dropout_backward(dd2, cache3)
  da1, dW2, db2 = affine_relu_backward(da2, cache2)
  dX, dW1, db1 = conv_relu_pool_backward(da1, cache1)

  grads = { 'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2, 'W3': dW3, 'b3': db3 }

  reg_loss = 0.0
  for p in ['W1', 'W2', 'W3']:
    W = model[p]
    reg_loss += 0.5 * reg * np.sum(W * W)
    grads[p] += reg * W
  loss = data_loss + reg_loss

  return loss, grads


def init_five_layer_convnet(input_shape=(3, 64, 64), num_classes=100,
                            filter_sizes=(5, 5, 5), num_filters=(32, 32, 64, 128),
                            weight_scale=1e-2, bias_scale=0, dtype=np.float32):
  """
  Initialize a five-layer convnet with the following architecture:

  [conv - relu - pool] x 3 - affine - relu - dropout - affine - softmax

  Each pooling region is 2x2 stride 2 and each convolution uses enough padding
  so that all convolutions are "same".

  Inputs:
  - Input shape: A tuple (C, H, W) giving the shape of each input that will be
    passed to the ConvNet. Default is (3, 64, 64) which corresponds to
    TinyImageNet.
  - num_classes: Number of classes over which classification will be performed.
    Default is 100 for TinyImageNet-100-A / TinyImageNet-100-B.
  - filter_sizes: Tuple of 3 integers giving the size of the filters for the
    three convolutional layers. Default is (5, 5, 5) which corresponds to 5x5
    filter at each layer.
  - num_filters: Tuple of 4 integers where the first 3 give the number of
    convolutional filters for the three convolutional layers, and the last
    gives the number of output neurons for the first affine layer.
    Default is (32, 32, 64, 128).
  - weight_scale: All weights will be randomly initialized from a Gaussian
    distribution whose standard deviation is weight_scale.
  - bias_scale: All biases will be randomly initialized from a Gaussian
    distribution whose standard deviation is bias_scale.
  - dtype: numpy datatype which will be used for this network. Float32 is
    recommended as it will make floating point operations faster.
  """
  C, H, W = input_shape
  F1, F2, F3, FC = num_filters
  filter_size = 5
  model = {}
  model['W1'] = np.random.randn(F1, 3, filter_sizes[0], filter_sizes[0])
  model['b1'] = np.random.randn(F1)
  model['W2'] = np.random.randn(F2, F1, filter_sizes[1], filter_sizes[1])
  model['b2'] = np.random.randn(F2)
  model['W3'] = np.random.randn(F3, F2, filter_sizes[2], filter_sizes[2])
  model['b3'] = np.random.randn(F3)
  model['W4'] = np.random.randn(H * W * F3 / 64, FC)
  model['b4'] = np.random.randn(FC)
  model['W5'] = np.random.randn(FC, num_classes)
  model['b5'] = np.random.randn(num_classes)

  for i in [1, 2, 3, 4, 5]:
    model['W%d' % i] *= weight_scale
    model['b%d' % i] *= bias_scale

  for k in model:
    model[k] = model[k].astype(dtype, copy=False)

  return model


def five_layer_convnet(X, model, y=None, reg=0.0, dropout=1.0,
                       extract_features=False, compute_dX=False,
                       return_probs=False):
  """
  Compute the loss and gradient for a five layer convnet with the architecture

  [conv - relu - pool] x 3 - affine - relu - dropout - affine - softmax

  Each conv is stride 1 with padding chosen so the convolutions are "same";
  all padding is 2x2 stride 2.

  We use L2 regularization on all weight matrices and no regularization on
  biases.

  This function can output several different things:

  If y not given, then this function will output extracted features,
  classification scores, or classification probabilities depending on the
  values of the extract_features and return_probs flags.

  If y is given, then this function will output either (loss, gradients)
  or dX, depending on the value of the compute_dX flag.

  Inputs:
  - X: Input data of shape (N, C, H, W)
  - model: Dictionary mapping string names to model parameters. We expect the
    following parameters:
    W1, b1, W2, b2, W3, b3: Weights and biases for the conv layers
    W4, b4, W5, b5: Weights and biases for the affine layers
  - y: Integer vector of shape (N,) giving labels for the data points in X.
    If this is given then we will return one of (loss, gradient) or dX;
    If this is not given then we will return either class scores or class
    probabilities.
  - reg: Scalar value giving the strength of L2 regularization.
  - dropout: The probability of keeping a neuron in the dropout layer

  Outputs:
  This function can return several different things, depending on its inputs
  as described above.

  If y is None and extract_features is True, returns:
  - features: (N, H) array of features, where H is the number of neurons in the
    first affine layer.
  
  If y is None and return_probs is True, returns:
  - probs: (N, L) array of normalized class probabilities, where probs[i][j]
    is the probability that X[i] has label j.

  If y is None and return_probs is False, returns:
  - scores: (N, L) array of unnormalized class scores, where scores[i][j] is
    the score assigned to X[i] having label j.

  If y is not None and compute_dX is False, returns:
  - (loss, grads) where loss is a scalar value giving the loss and grads is a
    dictionary mapping parameter names to arrays giving the gradient of the
    loss with respect to each parameter.

  If y is not None and compute_dX is True, returns:
  - dX: Array of shape (N, C, H, W) giving the gradient of the loss with
    respect to the input data.
  """
  W1, b1 = model['W1'], model['b1']
  W2, b2 = model['W2'], model['b2']
  W3, b3 = model['W3'], model['b3']
  W4, b4 = model['W4'], model['b4']
  W5, b5 = model['W5'], model['b5']

  conv_param_1 = {'stride': 1, 'pad': (W1.shape[2] - 1) / 2}
  conv_param_2 = {'stride': 1, 'pad': (W2.shape[2] - 1) / 2}
  conv_param_3 = {'stride': 1, 'pad': (W3.shape[2] - 1) / 2}
  pool_param = {'stride': 2, 'pool_height': 2, 'pool_width': 2}
  dropout_param = {'p': dropout}
  dropout_param['mode'] = 'test' if y is None else 'train'

  a1, cache1 = conv_relu_pool_forward(X, W1, b1, conv_param_1, pool_param)
  a2, cache2 = conv_relu_pool_forward(a1, W2, b2, conv_param_2, pool_param)
  a3, cache3 = conv_relu_pool_forward(a2, W3, b3, conv_param_3, pool_param)
  a4, cache4 = affine_relu_forward(a3, W4, b4)

  if extract_features:
    return a4.reshape((X.shape[0], 512))

  d4, cache5 = dropout_forward(a4, dropout_param)
  scores, cache6 = affine_forward(d4, W5, b5)

  if y is None:
    if return_probs:
      probs = np.exp(scores - np.max(scores, axis=1, keepdims=True))
      probs /= np.sum(probs, axis=1, keepdims=True)
      return probs
    else:
      return scores

  data_loss, dscores = softmax_loss(scores, y)
  dd4, dW5, db5 = affine_backward(dscores, cache6)
  da4 = dropout_backward(dd4, cache5)
  da3, dW4, db4 = affine_relu_backward(da4, cache4)
  da2, dW3, db3 = conv_relu_pool_backward(da3, cache3)
  da1, dW2, db2 = conv_relu_pool_backward(da2, cache2)
  dX, dW1, db1 = conv_relu_pool_backward(da1, cache1)

  if compute_dX:
    return dX

  grads = {
    'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2,
    'W3': dW3, 'b3': db3, 'W4': dW4, 'b4': db4,
    'W5': dW5, 'b5': db5,
  }

  reg_loss = 0.0
  for p in ['W1', 'W2', 'W3', 'W4', 'W5']:
    W = model[p]
    reg_loss += 0.5 * reg * np.sum(W)
    grads[p] += reg * W
  loss = data_loss + reg_loss

  return loss, grads


pass
