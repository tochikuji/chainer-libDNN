# example of Couvolutional AutoEncoder Visualization

import libdnn.visualizer as V
from libdnn import AutoEncoder
import chainer
import chainer.functions as F
import numpy
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_mldata


model = chainer.FunctionSet(
    enc_conv=F.Convolution2D(1, 64, 5, pad=2),
    dec_conv=F.Convolution2D(64, 1, 5, pad=2)
)


def forward(self, x, train):
    if train:
        x = F.dropout(x, ratio=0.4)

    h = F.sigmoid(self.model.enc_conv(x))
    h = self.model.dec_conv(h)

    return h

ae = AutoEncoder(model, gpu=-1)
ae.set_forward(forward)
ae.load_param('./cae.param.npy')

imager = V.Visualizer(ae)
imager.plot_filters('enc_conv')
imager.plot_filters('dec_conv')
mnist = fetch_mldata('MNIST original', data_home='.')
mnist.data = mnist.data.astype(numpy.float32) / 255


def output(self, x, layer):
    h = self.model.enc_conv(x)
    if layer == 1:
        return h
    h = F.sigmoid(h)
    h = self.model.dec_conv(h)
    if layer == 2:
        return h

    return None


ae.set_output(output)

for i in range(30):
    perm = numpy.random.permutation(70000)
    data = mnist.data[perm[0]]
    do = numpy.random.permutation(28 ** 2)[:int((28 ** 2) * 0.4)]
    data[do] = 0.0

    plt.imshow(mnist.data[perm[0]].reshape(28, 28), interpolation='none', cmap=mpl.cm.gray)
    plt.show()

    imager.plot_output(mnist.data[perm[0]].reshape(1, 1, 28, 28), layer=2)
    imager.plot_output(data.reshape(1, 1, 28, 28), layer=2)
