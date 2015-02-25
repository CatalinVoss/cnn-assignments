All models trained for 25 epochs on the tiny-imagenet-100-A dataset with a five
layer convnet architecture:

[conv - relu - pool] x 3 - affine - relu - affine - softmax

All conv layers are 3x3 stride 1, and all pool layers are 2x2 stride 2.
The first and second convolutional layer have 32 filters, and the third has 64
filters. The hidden affine layer has 512 neurons.
