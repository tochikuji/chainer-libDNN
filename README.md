# Chainer-libCNN
## Description
General utilities for deep neural network framework chainer.

It contains well-usable network training interface for general neural networks and visualizer class for Convolution neural network.

## Requirements
Minimum requirements:  

- Python 2.7 or 3.4 later (This has 2 and 3 compatibility)
- Chainer(>= 1.2.0) and minimum dependencies
- cv2: opencv python interface (Visualizer)
- matplotlib (Visualizer)

## Installation
`python setup.py install` 

## License
This software is released under MIT License.  
For more details about license, refer to 'LICENSE'.

# Usage
## Define networks
You have to define network structure and specify loss-function and optimizer method.
It will be ready for train.
```python
from libdnn import Classifier
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

# Generate instance (GPGPU enabled)
network = Classifier(model, gpu=0)
# Override instance forward method
network.set_forward(forward)

# Some operation to import data

# training
error_on_train, accuracy_on_train = network.train(train_data, target)

# test
error_on_test, accuracy_on_test = network(test_data, target)

print(accuracy_on_test)

# save trained network parameters
network.save_param('network.param.npy')
```

## Visualization
Visualizer class provides filter visualization for Convolution neural network(CNN).

```python
from libdnn import Classifier
import libdnn.Visualizer as V

# define CNN and train it
# SOME CODES

# define forwarding function that returns specified layer output
def output(self, x, layer):
    h = network.conv1(x)
    if layer == 1:
        return h

    h = F.max_pooling_2d(F.relu(h), 2)
    h = ...

network.set_output(output)

# instance
imager = V.Visualizer(network)

# view all filters on matplotlib.pyplot.imshow
imager.plot_filters('conv1')
# write filter to image file
imager.write_filters('conv1', path='./filter_img', identifier='img_', type='bmp')

# view layer activation by all filters
imager.plot_output(data[:2], layer=1)
# write activation to image files
imager.write_output(data, layer=2, path='./output_img', identifier='l2_', type='bmp')
```

# References
## Classifier, Regressor, AutoEncoder
There are 3 types of network templates according to neural network object, Classification, Regression, and AutoEncoder tasks.  
Each class has almost same interfaces like below.  
For more detailed usage, please refer to examples.

#### Constractor / Initializer
`__init__(self, model, gpu=-1)`  
1 argument `model` is neccesary.  
`model` expects instance of `chainer.FunctionSet` (Neural network structure)  
`gpu` is optional. if `gpu >= 0` is True, most network culculation will be processed by CUDA computation.
If you want to use GPGPU, set `gpu` as your CUDA device ID.

#### Feedforwarding method
`@abstructmethod forward(self, x, train)`  
It is pure virtual function, you must override this method on instance or subclass/derivered class by `set_forward`.  
3 arguments needed, `self` is a magic variable, `x`, is `chainer.Variable` instance and `train` is training stage flag (some networks e.g. AutoEncoder needed is this is training input or not).  
Define feedforward Computation as you like. and it must return output of neural network(isa `chainer.Variable`).  
You needn't to care about `gpu`, it will be automatically converted properly.

#### Override forward method
`set_forward(self, func)`  
This method will override `CNNBase.forward` as specified feedforward function.  
1 argument, isa `function` is needed.

#### See specified layer activations
`@abstructmethod output(self, x, layer)`  
It it also pure virtual function, that must be Overridden by `set_output`. It is only used for visualization layer output images in `Visualizer` class.
If you won't use these features, it will be unnecessary.  
3 arguments is needed, `self` is a magic variable, x, isa `chainer.Variable` instance, layer is a layer specification flag (that you use in your implementation).  
This will be similar to `forward` function in most case. But do *NOT* reuse as `forward` function because of interference CUDA optimization.

#### Override forward method
`set_forward(self, func)`  
This method will override `CNNBase.output` as specified feedforward function.  
1 argument, isa `function` is needed. You needn't to care about use GPU whether or not.

#### Configure loss_function
`set_loss_function(self, func, param={})`  
1 argument `func` is neccesary.  
This expects loss function object in `chainer.functions`.  
`param` is optional. This expects dictionary that specifies loss function parameters.  
If you didn't specify loss function by this method, it will be set in network definition automatically. 

#### Configure optimizer
`set_optimizer(self, optimizer, param={})`  
1 argument `optimizer` is neccesary.  
This expects optimizer function object in `chainer.optimiers`.  
`param` is optional. This expects dictionary that specifies optimizer parameter like `{lr: 0.01, rho: 0.9}`.  
If you didn't specify optimizer function by this method, in most cases, network will use `chainer.optimizers.Adam` by default.

#### Train network
`train(self, x_data, y_data, batchsize=100, action=(lambda: None))`  
2 arguments `x_data, y_data` are necessary.  
`x_data` : training data, `y_data`: target value, isa numpy.array;  
`batchsize` is a number of minibatch training.(defalut 100)  
`action` expects function object. This will be called on end of  minibatch training each time.  

It returns error and accuracy ratio for training data.
#### Test/Validate network
`test(self, x_data, y_data, action=(lambda: None))`  
It returns error and accuracy ratio for test data to validate training result.

#### Save trained network parameters to file
`save_param(self, dst)`  
1 argument `dst` is optional that specifies destination.  
It will be './network.param.npy' by default.

#### Load trained network parameters from file
`load_param(self, src)`  
1 argument `src` is optional that specifies source file.  
It will be './network.param.npy' same as `save_param` by default.

## Visualizer
#### Constructor / Initializer
`Visualizer.__init__(self, model)`  
1 argument `model` is necessary.  
`model` expects `CNNBase` instance that is defined as CNN. (no longer CNNBase.model in any cases)  

#### Convert filters to image data
`Visualizer.convert_filters(self, layer, height, width)`  
DO call this method before other `~~_filters` methods.  
1 argument `layer` is necessary. `layer` expects a name of layer that you declared network construction.  
height, width is optional. If you didn't specify, filter size will be auto-detected(same as kernel size).
In most cases, you needn't specify these value explicitly.

#### View filters on matplotliib
`Visualizer.plot_filters(self, layer, shape, T, title)`  
View all filters on matplotlib.
This will break program running by default.  
1 argument `layer` is neccesary. `layer` expects a name of layer that you declared network definition.  
`shape` and `T` are optinal. They will be required by needs of fully-connected layer Visualization 
(e.g. fully-connected autoencoder).  
`shape` expect tuple object. It is a filter size specification like `(height, width)`.  
And if `T` is True, layer weights will transposed (for decoding layers).  
`title` is optional. By default, each filter has a title like filter\_xxx, 
but it often overlap on another filter images.  
If you needn't to show titles, set `T=False`.  

#### Write filters to image file
`Visualizer.write_filters(self, layer, path='./', identifier='img', type='bmp', shape, T)`  
1 argument `layer` is necessary, same as `plot_filters`.  
`path` is a path to store images. It will store current directory by default.  
`identifier` is a identifier(prefix) of image files. like 'idenfier_00xxx.jpg'  
`type` is a image data format. It will be Windows Bitmap Image format by default.  
We recommend you to use uncompressed formats(e.g. bmp, tiff, pgm etc.)  
`shape, T` are same as `plot_filters`.

#### Write filters on CSV
`Visualizer.save_raw_filter(self, dst)`  
For maestri of MATLAB.  
1 argument `dst`, a path to save dir is neccesary.  
And also need `convert_filters` in advance.

#### View layer activations on matplotlib
`Visualizer.plot_output(self, x, layer)`  
View layer activations on (trained) network with matplotlib for inputs `x`.
2 arguments is neccesary.  
`x` is a images that you want to see derivered one.  
`layer` is a layer specification flag that you used on `output` method.  
This shows all filtered images (number of images: x.shape[0]) * (number of output channels). Watch out for it.

#### Write layer activations to image files
`Visualizer.write_output(self, x, layer, path, identifier, type)`  
2 arguments, `x`, `layer` are necessary same as `plot_output` above.  
`path` is a path to store images. It will store current directory by default.  
`identifier` is a identifier(prefix) of image files. like 'idenfier\_(imagenum)\_f(filternum).jpg'  
`type` is a image data format. It will be Windows Bitmap Image format by default.  

#### Write layer activation for an image
`Visualizer.write_activation(self, x, layer, path, identifier, type)`  
It has only one difference from `write_output` that is source image would be expected only one. (shape dim = 3)  
It requires parameters same as `write_output`.

## Author
Aiga SUZUKI\<ai-suzuki@aist.go.jp, tochikuji@gmail.com>
