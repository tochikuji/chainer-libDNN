# example of Convolutional Auto-encoder with layer visualization

from libdnn import AutoEncoder
import libdnn.visualizer as V
import chainer
import chainer.functions as F
import chainer.optimizers as Opt
import numpy
from sklearn.datasets import fetch_mldata


model = chainer.FunctionSet(
    enc_conv=F.Convolution2D(1, 128, 3, pad=1),
    # fh1=F.Linear(1000, 4096),
    # fh2=F.Linear(4096, 4096),
    # fh3=F.Linear(4096, 1000),
    dec_conv=F.Convolution2D(128, 1, 3, pad=1)
)


def forward(self, x, is_train):
    if is_train:
        F.dropout(x, ratio=0.3)

    h = F.tanh(self.model.enc_conv(x))
    h = F.tanh(self.model.dec_conv(h))

    return h

ae = AutoEncoder(model, gpu=0)
ae.set_forward(forward)

mnist = fetch_mldata('MNIST original', data_home='.')
perm = numpy.random.permutation(len(mnist.data))
train_data = mnist.data[perm][:60000].astype(numpy.float32).reshape(60000, 1, 28, 28)
test_data = mnist.data[perm][60000:].astype(numpy.float32).reshape(10000, 1, 28, 28)

for epoch in range(50):
    print('epoch : %d' % epoch)
    err = ae.train(train_data)
    print(err)
    err = ae.test(test_data)
    print(err)
