# example of Convolutional Auto-encoder with layer visualization

from libdnn import AutoEncoder
import chainer
import chainer.functions as F
import chainer.optimizers as Opt
import numpy
from sklearn.datasets import fetch_mldata


model = chainer.FunctionSet(
    fh1=F.Linear(28 ** 2, 200),
    fh3=F.Linear(200, 28 ** 2),
)


def forward(self, x, train):
    if train:
        x = F.dropout(x, ratio=0.4)

    h = F.dropout(F.sigmoid(self.model.fh1(x)), train=train)
    h = F.dropout(self.model.fh3(h), train=train)

    return h

ae = AutoEncoder(model, gpu=0)
ae.set_forward(forward)
ae.set_optimizer(Opt.Adam)

mnist = fetch_mldata('MNIST original', data_home='.')
perm = numpy.random.permutation(len(mnist.data))
mnist.data = mnist.data.astype(numpy.float32) / 255
train_data = mnist.data[perm][:60000]
test_data = mnist.data[perm][60000:]

for epoch in range(300):
    print('epoch : %d' % (epoch + 1))
    err = ae.train(train_data, batchsize=200)
    print(err)
    perm = numpy.random.permutation(len(test_data))
    terr = ae.test(test_data[perm][:100])
    print(terr)

    with open('ae.log', mode='a') as f:
        f.write("%d %f %f\n" % (epoch + 1, err, terr))

ae.save_param('ae.param.npy')
