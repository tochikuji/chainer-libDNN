# Chainer-libCNN
## Description
General utilities for deep neural network framework chainer.

It contains well-usable network training interface for general neural networks and visualizer class for Convolution neural network.

# Usage
## Define networks
You have to define network structure and specify loss-function and optimizer method.
It will be ready for train.
```python
import CNN
import chainer
import chainer.functions as F
import chainer.optimizers as Opt


# declare network structure
model = chainer.FunctionSet(
	l1 = .....,
    # Specify layer construction as you like
)

# define feedforward calculation
def forward(self, x):
	h = F.relu(self.model.l1(x))
    ...
    y = self.model.Lout(h)
    
    return y

# Generate instance
network = CNN.CNNBase(model, is_gpu=0)
# Override instance forward method
network.forward = forward
# Set loss function and optimizer function
network.set_optimizer(loss_function=F.softmax_cross_entropy, optimizer=Opt.Adam)

# Some operation for import datas

# training
error_on_trian, accuracy_on_train = network.train(train_data, target)

# test
error_on_test, accuracy_on_test = network(test_data, target)

print(accuracy_on_test)
```

## Visualization
Visualizer class provides filter visualization for Convolution neural network(CNN).
```python
import CNN
import Visualizer as V

# define CNN and train it
# SOME CODES

# instance
imager = V.Visualizer(network.model, 'conv1')
# generate filter matrix
imager.convert_filters(height=5, width=5)

# view all filters on matplotlib.pyplot.imshow
imager.plot_filters()
# write filter to image file
imager.write_filters(path='./filter_img', identifier='img_', type='bmp')
```

# References
## CNNBase
#### Constractor / Initializer
`CNNBase.__init__(self, model, is_gpu=-1)`  
1 argument `model` is neccesary.  
`model` expects instance of `chainer.FunctionSet` (Neural network structure)  
`is_gpu` is optional. if `is_gpu >= 0` is True, most network culculation will be processed by CUDA computation.
If you want to use GPGPU, set `is_gpu` as your CUDA device ID.

#### Define forward method
`@abstructmethod forward(self, x)`  
It is pure virtual function, you must override this method on instance or subclass/derivered class.  
2 arguments needed, `self` is a magic variable, and x, is `chainer.Variable` instance.  
Define feedforward Computation as you like. and it must return output of neural network(isa `chainer.Variable`).  
You needn't to care about `is_gpu`, it will be automatically converted properly.

#### Configure optimizer
`set_optimizer(loss_function, optimizer=Opt.Adam)`  
1 argument loss_function is neccesary.  
`loss_function` expects `chainer.function` instance which is function object.  
`optimizer` expects optimizer function. If you ignore this argument, this will default to  `chainer.optimizers.Adam`.

#### Train
`train(self, x_data, y_data, batchsize=100, action=(lambda: None))`  
2 arguments `x_data, y_data` are necessary.  
`x_data` : training data, `y_data`: target value, isa numpy.array;  
`batchsize` is a number of minibatch training.(defalut 100)  
`action` expects function object. This will be called on end of  minibatch training each time.  

It returns error and accuracy ratio for training data.
#### Test
`test(self, x_data, y_data, action=(lambda: None))`  
It returns error and accuracy ratio for test data to validate training result.

## Visualizer
#### Constructor / Initializer
`Visualizer.__init__(self, model, layer)`  
2 arguments `model, layer` are necessary.  
`model` expects `chainer.FunctionSet` instance. In most cases, it will be `CNNBase.model`;  
`layer` expects a name of layer that you declared network construction.  

#### Convert filters to image data
`Visualizer.convert_filters(self, height, width)`  
DO call this method before other method.  
height, width is optional. If you didn't specify, filter size will be auto-detected(same as kernel size).

#### View on matplotliib
`Visualizer.plot_filters(self)`  
View all filters on matplotlib. It will break program running.

#### Write filter image to file
`Visualizer.write_filters(self, path='./', identifier='img', type='bmp')`  
All arguments are optional.  
`path` is a path to store images. It will store current directory by default.  
`identifier` is a identifier(prefix) of image files. like 'idenfier_00xxx.jpg'  
`type` is a image data format. It will be Windows Bitmap Image format by default.  
We recommend you to use uncompressed formats(e.g. bmp, tiff, pgm etc.)
