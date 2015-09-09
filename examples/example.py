# CNN module example of XOR classification with Multi Layer Perceptron
import sys

sys.path.append('..')

import chainer
import numpy
import chainer.functions as F
import chainer.optimizers as Opt
from libdnn import Classifier


# multi layer perceptron
model = chainer.FunctionSet(l1=F.Linear(2, 100), l2=F.Linear(100, 100), l3=F.Linear(100, 2))


# define forwarding method
def forward(self, x, train):
    h = F.relu(self.model.l1(x))
    h = F.relu(self.model.l2(h))
    y = self.model.l3(h)

    return y

mlp = Classifier(model, gpu=-1)
mlp.set_forward(forward)
mlp.set_optimizer(Opt.AdaDelta, {'rho': 0.9})

arr = []
t = []
for i in range(10000):
    x, y = (numpy.random.rand() - 0.5), (numpy.random.rand() - 0.5)
    arr.append(numpy.array([x, y]))
    if (x < 0. and y < 0.) or (x > 0. and y > 0.):
        t.append(0)
    else:
        t.append(1)

print(mlp.train(numpy.array(arr).astype(numpy.float32), numpy.array(t).astype(numpy.int32)))
print(mlp.test(numpy.array(arr).astype(numpy.float32), numpy.array(t).astype(numpy.int32)))
