# example of Convolutional Auto-encoder

from libdnn import AutoEncoder
import libdnn.visualizer as V
import chainer
import chainer.functions as F
import chainer.optimizers as Opt
import numpy
from sklearn.datasets import fetch_mldata
import matplotlib.pyplot as plt


model = chainer.FunctionSet(
    enc_conv=F.Convolution2D(1, 64, 5, pad=2),
    dec_conv=F.Convolution2D(64, 1, 5, pad=2)
)


def forward(self, x, train):
    if train:
        x = F.dropout(x, ratio=0.4, train=train)

    h = F.sigmoid(self.model.enc_conv(x))
    h = self.model.dec_conv(h)

    return h

ae = AutoEncoder(model, gpu=-1)
ae.set_forward(forward)

mnist = fetch_mldata('MNIST original', data_home='.')
perm = numpy.random.permutation(len(mnist.data))
mnist.data = mnist.data.astype(numpy.float32) / 255
train_data = mnist.data[perm][:60000].reshape(60000, 1, 28, 28)
test_data = mnist.data[perm][60000:].reshape(10000, 1, 28, 28)

for epoch in range(40):
    print('epoch : %d' % epoch)
    err = ae.train(train_data, batchsize=200)
    print(err)
    perm = numpy.random.permutation(len(test_data))
    terr = ae.test(test_data[perm][:100])
    print(err)
    with open('test.log', mode='a') as f:
        f.write("%d %f %f\n" % (epoch + 1, err, terr))

ae.save_param('./cae.param.npy')
