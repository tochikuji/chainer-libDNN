# example of Couvolutional AutoEncoder Visualization

import libdnn.visualizer as V
from libdnn import Classifier
import chainer
import chainer.functions as F


model = chainer.FunctionSet(
    conv1=F.Convolution2D(1, 15, 5),
    bn1=F.BatchNormalization(15),
    conv2=F.Convolution2D(15, 30, 3, pad=1),
    bn2=F.BatchNormalization(30),
    conv3=F.Convolution2D(30, 64, 3, pad=1),
    fl4=F.Linear(2304, 576),
    fl5=F.Linear(576, 10)
)


def forward(self, x, train):
    h = F.max_pooling_2d(F.relu(model.bn1(model.conv1(x))), 2)
    h = F.relu(model.bn2(model.conv2(h)))
    h = F.max_pooling_2d(F.relu(model.conv3(h)), 2)
    h = F.dropout(F.relu(model.fl4(h)), train=True)
    y = model.fl5(h)

    return y


cnn = Classifier(model, gpu=-1)
cnn.set_forward(forward)
cnn.load_param('./cnn.param.npy')

imager = V.Visualizer(cnn)
imager.plot_filters('conv1', interpolation=True)
imager.plot_filters('conv2', title=False, interpolation=True)
imager.plot_filters('conv3', title=False, interpolation=True)
