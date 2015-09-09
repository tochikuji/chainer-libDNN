# CNN class example of Convolution Neural Network to ramdom datas.
# This is also example of Visualizer module

import sys

sys.path.append('../')

import chainer
import numpy
import chainer.functions as F
import libdnn.visualizer as V
from libdnn import Classifier


# define 3-layer convolutional neural network
model = chainer.FunctionSet(
    conv1=F.Convolution2D(1, 15, 5),
    bn1=F.BatchNormalization(15),
    conv2=F.Convolution2D(15, 30, 3, pad=1),
    bn2=F.BatchNormalization(30),
    conv3=F.Convolution2D(30, 64, 3, pad=1),
    fl4=F.Linear(576, 576),
    fl5=F.Linear(576, 10)
)


def forward(self, x, train):
    h = F.max_pooling_2d(F.relu(model.bn1(model.conv1(x))), 2)
    h = F.max_pooling_2d(h, 2)
    h = F.max_pooling_2d(F.relu(model.bn2(model.conv2(h))), 2)
    h = F.max_pooling_2d(F.relu(model.conv3(h)), 2)
    h = F.dropout(F.relu(model.fl4(h)), train=True)
    y = model.fl5(h)

    return y

cnn = Classifier(model, gpu=-1)
cnn.set_forward(forward)

arr = []
t = []
for i in range(100):
    x = numpy.random.rand(50, 50)
    x = numpy.array([x])
    arr.append(x)
    a = numpy.random.randint(0, 9)
    t.append(a)

print(cnn.train(numpy.array(arr).astype(numpy.float32), numpy.array(t).astype(numpy.int32)))
print(cnn.test(numpy.array(arr).astype(numpy.float32), numpy.array(t).astype(numpy.int32)))


def output(self, x, layer):
    h = model.bn1(model.conv1(x))
    if layer == 1:
        return h

    h = F.max_pooling_2d(F.relu(h), 2)
    h = model.bn2(model.conv2(h))
    if layer == 2:
        return h

    h = F.max_pooling_2d(F.relu(h), 2)
    h = model.conv3(h)
    if layer == 3:
        return h

    return None


cnn.set_output(output)
imager = V.Visualizer(cnn)
imager.plot_filters('conv1')
imager.write_filters('conv1', path='./filter', identifier='filter_', type='bmp')

imager.plot_output(numpy.array(arr).astype(numpy.float32)[:3], layer=1)
imager.write_output(numpy.array(arr).astype(numpy.float32), layer=2, path='./outputs', identifier='l1_', type='bmp')
imager.write_activation(arr[0], 2)

cnn.save_param()
