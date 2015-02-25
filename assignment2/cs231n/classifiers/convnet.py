import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


def two_layer_convnet(X, model, y=None, reg=0.0):
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

  # Compute the forward pass
  a1, cache1 = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
  scores, cache2 = affine_forward(a1, W2, b2)

  if y is None:
    return scores

  # Compute the backward pass
  data_loss, dscores = softmax_loss(scores, y)

  # Compute the gradients using a backward pass
  da1, dW2, db2 = affine_backward(dscores, cache2)
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

def init_supercool_convnet_big(weight_scale=5e-2, bias_scale=0, input_shape=(3, 32, 32),
                           num_classes=10, num_filters=32, filter_size=3):
  C, H, W = input_shape
  assert filter_size % 2 == 1, 'Filter size must be odd; got %d' % filter_size

  model = {}
  model['W1'] = weight_scale * np.random.randn(num_filters, C, filter_size, filter_size)
  model['W2'] = weight_scale * np.random.randn(num_filters, num_filters, filter_size, filter_size)
  model['W3'] = weight_scale * np.random.randn(num_filters, num_filters, filter_size, filter_size)
  model['W4'] = weight_scale * np.random.randn(i) # after pooling, size is less than H*W, divided by 4
  model['W5'] = weight_scale * np.random.randn(num_filters * H * W / 8, num_classes)

  model['b1'] = bias_scale * np.random.randn(num_filters)
  model['b2'] = bias_scale * np.random.randn(num_filters)
  model['b3'] = bias_scale * np.random.randn(num_filters)
  model['b4'] = bias_scale * np.random.randn(num_filters * H * W / 8)
  model['b5'] = bias_scale * np.random.randn(num_classes)
  return model


def supercool_convnet_big(X, model, y=None, reg=1e-4):
  # 1 conv_relu_forward
  # 2 conv_relu_pool_forward
  # 3 affine_forward
  # 4 affine_forward -- becomes SVM
  W1 = model['W1']
  W2 = model['W2']
  W3 = model['W3']
  W4 = model['W4']
  W4 = model['W5']
  b1 = model['b1']
  b2 = model['b2']
  b3 = model['b3']
  b4 = model['b4']
  b4 = model['b5']

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

  # Forward
  o1, cache1 = conv_relu_forward(X, W1, b1, conv_param)
  o2, cache2 = conv_relu_pool_forward(o1, W2, b2, conv_param, pool_param)
  o2, cache2 = conv_relu_pool_forward(o1, W2, b2, conv_param, pool_param)
  o3, cache3 = affine_forward(o2, W3, b3)
  scores, cache4 = affine_forward(o3, W4, b4)

  if y is None:
    return scores

  # Compute the backward pass
  # Make last layer linear SVM
  data_loss, dscores = softmax_loss(scores, y)

  # Compute the gradients using a backward pass
  do3, dW4, db4 = affine_backward(dscores, cache4)
  do2, dW3, db3 = affine_backward(do3, cache3)
  do1, dW2, db2 = conv_relu_pool_backward(do2, cache2)
  dX,  dW1, db1 = conv_relu_backward(do1, cache1)

  # Add regularization
  dW1 += reg * W1
  dW2 += reg * W2
  dW3 += reg * W3
  dW4 += reg * W4
  reg_loss = 0.5 * reg * sum(np.sum(W * W) for W in [W1, W2, W3, W4])

  loss = data_loss + reg_loss
  grads = {'W1': dW1, 'W2': dW2, 'W3': dW3, 'W4': dW4, 'b1': db1, 'b2': db2, 'b3': db3, 'b4': db4}
  
  return loss, grads


def init_supercool_convnet(weight_scale=5e-2, bias_scale=0, input_shape=(3, 32, 32),
                           num_classes=10, num_filters=32, filter_size=3):
  C, H, W = input_shape
  assert filter_size % 2 == 1, 'Filter size must be odd; got %d' % filter_size

  model = {}
  model['W1'] = weight_scale * np.random.randn(num_filters, C, filter_size, filter_size)
  model['W2'] = weight_scale * np.random.randn(num_filters, num_filters, filter_size, filter_size)
  # /4 /8 if pooling once, /16, /32 if pooling twice
  # /16 /64 does better; less params at end
  model['W3'] = weight_scale * np.random.randn(num_filters * H * W / 16, num_filters * H * W / 64) # after pooling, size is less than H*W, divided by 4
  model['W4'] = weight_scale * np.random.randn(num_filters * H * W / 64, num_classes)

  model['b1'] = bias_scale * np.random.randn(num_filters)
  model['b2'] = bias_scale * np.random.randn(num_filters)
  model['b3'] = bias_scale * np.random.randn(num_filters * H * W / 64)
  model['b4'] = bias_scale * np.random.randn(num_classes)
  return model


def supercool_convnet(X, model, y=None, reg=1e-4):
  # 1 conv_relu_forward
  # 2 conv_relu_pool_forward
  # 3 affine_forward
  # 4 affine_forward -- becomes SVM
  W1 = model['W1']
  W2 = model['W2']
  W3 = model['W3']
  W4 = model['W4']
  b1 = model['b1']
  b2 = model['b2']
  b3 = model['b3']
  b4 = model['b4']

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

  # Forward
  o1, cache1 = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
  o2, cache2 = conv_relu_pool_forward(o1, W2, b2, conv_param, pool_param)
  o3, cache3 = affine_forward(o2, W3, b3)
  o4, cache5 = relu_forward(o3)
  scores, cache4 = affine_forward(o4, W4, b4)

  if y is None:
    return scores

  # Compute the backward pass
  # Make last layer linear SVM
  data_loss, dscores = svm_loss(scores, y)

  # Compute the gradients using a backward pass
  do4, dW4, db4 = affine_backward(dscores, cache4)
  do3 = relu_backward(do4, cache5)
  do2, dW3, db3 = affine_backward(do3, cache3)
  do1, dW2, db2 = conv_relu_pool_backward(do2, cache2)
  dX,  dW1, db1 = conv_relu_pool_backward(do1, cache1)

  # Add regularization
  dW1 += reg * W1
  dW2 += reg * W2
  dW3 += reg * W3
  dW4 += reg * W4
  reg_loss = 0.5 * reg * sum(np.sum(W * W) for W in [W1, W2, W3, W4])

  loss = data_loss + reg_loss
  grads = {'W1': dW1, 'W2': dW2, 'W3': dW3, 'W4': dW4, 'b1': db1, 'b2': db2, 'b3': db3, 'b4': db4}
  
  return loss, grads

