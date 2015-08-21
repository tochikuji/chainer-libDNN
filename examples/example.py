import CNN
import chainer
import numpy
import chainer.functions as F
import chainer.optimizers as Opt


model = chainer.FunctionSet(l1=F.Linear(2, 100), l2=F.Linear(100, 100), l3=F.Linear(100, 2))


def forward(self, x):
    h = F.relu(self.model.l1(x))
    h = F.relu(self.model.l2(h))
    y = self.model.l3(h)

    return y

mlp = CNN.CNNBase(model, is_gpu=-1)
mlp.forward = forward
mlp.SetOptimizer(loss_function=F.softmax_cross_entropy, optimizer=Opt.Adam)

arr = []
t = []
for i in range(1000):
    x, y = (numpy.random.rand() - 0.5), (numpy.random.rand() - 0.5)
    arr.append(numpy.array([x, y]))
    if (x < 0. and y < 0.) or (x > 0. and y > 0.):
        t.append(0)
    else:
        t.append(1)

print(mlp.train(numpy.array(arr).astype(numpy.float32), numpy.array(t).astype(numpy.int32)))
print(mlp.test(numpy.array(arr).astype(numpy.float32), numpy.array(t).astype(numpy.int32)))
