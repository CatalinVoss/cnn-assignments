import numpy as np
import matplotlib.pyplot as plt
from cs231n.classifier_trainer import ClassifierTrainer
from cs231n.gradient_check import eval_numerical_gradient
from cs231n.classifiers.convnet import *
import cPickle

print 'running'

def rel_error(x, y):
    """ returns relative error """
    return np.max(np.abs(x - y) / (np.maximum(1e-18, np.abs(x) + np.abs(y))))

from cs231n.data_utils import load_CIFAR10

def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=1000):
    """
    Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
    it for the two-layer neural net classifier. These are the same steps as
    we used for the SVM, but condensed to a single function.  
    """
    # Load the raw CIFAR-10 data
    cifar10_dir = 'cs231n/datasets/cifar-10-batches-py'
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)
        
    # Subsample the data
    mask = range(num_training, num_training + num_validation)
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = range(num_training)
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = range(num_test)
    X_test = X_test[mask]
    y_test = y_test[mask]

    # Normalize the data: subtract the mean image
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image
    
    # Transpose so that channels come first
    X_train = X_train.transpose(0, 3, 1, 2).copy()
    X_val = X_val.transpose(0, 3, 1, 2).copy()
    x_test = X_test.transpose(0, 3, 1, 2).copy()

    return X_train, y_train, X_val, y_val, X_test, y_test

# Invoke the above function to get our data.
X_train, y_train, X_val, y_val, X_test, y_test = get_CIFAR10_data()
print 'Train data shape: ', X_train.shape
print 'Train labels shape: ', y_train.shape
print 'Validation data shape: ', X_val.shape
print 'Validation labels shape: ', y_val.shape
print 'Test data shape: ', X_test.shape
print 'Test labels shape: ', y_test.shape

# Loss check
model = init_supercool_convnet(weight_scale = 5e-2)
X = np.random.randn(100, 3, 32, 32)
y = np.random.randint(10, size=100)
loss, _ = supercool_convnet(X, model, y, reg=0)
# Sanity check: Loss should be about log(10) = 2.3026
print 'Sanity check loss (no regularization): ', loss
# Sanity check: Loss should go up when you add regularization
loss, _ = supercool_convnet(X, model, y, reg=1)
print 'Sanity check loss (with regularization): ', loss

# # Gradient check
# num_inputs = 2
# input_shape = (3, 16, 16)
# reg = 0.0
# num_classes = 10
# X = np.random.randn(num_inputs, *input_shape)
# y = np.random.randint(num_classes, size=num_inputs)
# model = init_supercool_convnet(num_filters=3, filter_size=3, input_shape=input_shape)
# loss, grads = supercool_convnet(X, model, y)
# for param_name in sorted(grads):
#     f = lambda _: supercool_convnet(X, model, y)[0]
#     param_grad_num = eval_numerical_gradient(f, model[param_name], verbose=False, h=1e-6)
#     e = rel_error(param_grad_num, grads[param_name])
#     print '%s max relative error: %e' % (param_name, rel_error(param_grad_num, grads[param_name]))

# # Make sure we can overfit...
# model = init_supercool_convnet(weight_scale=5e-2, bias_scale=0, filter_size=3) # weight_scale=5e-2
# trainer = ClassifierTrainer()
# best_model, loss_history, train_acc_history, val_acc_history = trainer.train(
#           X_train[:50], y_train[:50], X_val, y_val, model, supercool_convnet,
#           reg=0.001, momentum=0.9, learning_rate=0.0001, batch_size=10, num_epochs=10, # change to 20 epochs
#           verbose=True) # batch size 40-100

model = init_supercool_convnet(weight_scale=3e-2, bias_scale=0, filter_size=3)
trainer = ClassifierTrainer()
best_model, loss_history, train_acc_history, val_acc_history = trainer.train(
          X_train, y_train, X_val, y_val, model, supercool_convnet,
          reg=0.5, momentum=0.9, learning_rate=5e-5, batch_size=50, num_epochs=15, # change to 20 epochs
          verbose=True, acc_frequency=50) # batch size 40-100


with open('best_model_2.pkl', 'wb') as f:
    cPickle.dump(best_model, f)

# with open('best_model.pkl', 'rb') as f:
#     best_model = cPickle.load(f)
	