# CNN class example of Convolution Neural Network to ramdom datas.
# This is also example of Visualizer module
import sys

sys.path.append('..')

import CNN
import chainer
import numpy
import chainer.functions as F
import chainer.optimizers as Opt
import Visualizer


model = chainer.FunctionSet(
    conv1=F.Convolution2D(1, 30, 5),
    bn1=F.BatchNormalization(30),
    conv2=F.Convolution2D(30, 30, 3, pad=1),
    bn2=F.BatchNormalization(30),
    conv3=F.Convolution2D(30, 64, 3, pad=1),
    fl4=F.Linear(576, 576),
    fl5=F.Linear(576, 10)
)


def forward(self, x):
    h = F.max_pooling_2d(F.relu(model.bn1(model.conv1(x))), 2)
    h = F.max_pooling_2d(h, 2)
    h = F.max_pooling_2d(F.relu(model.bn2(model.conv2(h))), 2)
    h = F.max_pooling_2d(F.relu(model.conv3(h)), 2)
    h = F.dropout(F.relu(model.fl4(h)), train=True)
    y = model.fl5(h)

    return y

cnn = CNN.CNNBase(model, is_gpu=0)
cnn.forward = forward
cnn.setOptimizer(loss_function=F.softmax_cross_entropy, optimizer=Opt.Adam)

arr = []
t = []
for i in range(1000):
    x = numpy.random.rand(50, 50)
    x = numpy.array([x])
    arr.append(x)
    a = numpy.random.randint(0, 9)
    t.append(a)

print(cnn.train(numpy.array(arr).astype(numpy.float32), numpy.array(t).astype(numpy.int32)))
print(cnn.test(numpy.array(arr).astype(numpy.float32), numpy.array(t).astype(numpy.int32)))


def output(self, x, layer):
    h = F.relu(model.bn1(model.conv1(x)))
    if layer == 1:
        return h

    h = F.max_pooling_2d(h, 2)
    h = F.relu(model.bn2(model.conv2(h)))
    if layer == 2:
        return h

    h = F.max_pooling_2d(h, 2)
    h = F.relu(model.conv3(h))
    if layer == 3:
        return h

    return None


imager = Visualizer.Visualizer(cnn.model, 'conv1')
imager.convert_filters()
imager.plot_filters()
imager.write_filters(path='./filter', identifier='filter_', type='bmp')
