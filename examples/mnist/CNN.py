# example of Convolutional Neural Network

from libdnn import Classifier
import chainer
import chainer.functions as F
import numpy
from sklearn.datasets import fetch_mldata

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

mnist = fetch_mldata('MNIST original', data_home='.')
perm = numpy.random.permutation(len(mnist.data))
mnist.data = mnist.data.astype(numpy.float32).reshape(70000, 1, 28, 28) / 255
mnist.target = mnist.target.astype(numpy.int32)
train_data = mnist.data[perm][:60000]
train_label = mnist.target[perm][:60000]
test_data = mnist.data[perm][60000:]
test_label = mnist.target[perm][60000:]

for epoch in range(15):
    print('epoch : %d' % (epoch + 1))
    err, acc = cnn.train(train_data, train_label, batchsize=200)
    print(err)
    perm = numpy.random.permutation(len(test_data))
    terr, tacc = cnn.test(test_data[perm][:100], test_label[perm][:100])
    print(terr)

    with open('cnn.log', mode='a') as f:
        f.write("%d %f %f\n" % (epoch + 1, err, terr))

cnn.save_param('cnn.param.npy')
