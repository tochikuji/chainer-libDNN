# coding: utf-8
# Simple chainer interfaces for Deep learning researching
# Author: Aiga SUZUKI <ai-suzuki@aist.go.jp>
from abc import abstractmethod
import chainer
import numpy
import chainer.functions as F
import chainer.optimizers as Opt


class CNNBase(object):
    def __init__(self, model, is_gpu=-1):
        self.model = model
        self.is_gpu = is_gpu

    # 抽象メソッド
    # インスタンス・子クラスで再定義のこと
    @abstractmethod
    def forward(self, x):
        pass

    def validate(self, x_data, y_data, train=True):
        x, t = chainer.Variable(x_data, volatile=not train), chainer.Variable(y_data, volatile=not train)
        h = self.forward(self, x)

        return self.loss_function(h, t), F.accuracy(h, t)

    def SetOptimizer(self, loss_function, optimizer=Opt.Adam):
        if self.is_gpu >= 0:
            chainer.cuda.init(self.is_gpu)
            self.model = self.model.to_gpu()
        self.optimizer = optimizer()
        self.loss_function = loss_function

        self.optimizer.setup(self.model.collect_parameters())


    def train(self, x_data, y_data, batchsize=100, action=(lambda: None)):
        # num of x_data
        N = len(x_data)
        perm = numpy.random.permutation(N)

        sum_accuracy = 0.
        sum_error = 0.

        for i in range(0, N, batchsize):
            # training mini batch of x, y
            x_batch = x_data[perm[i:i + batchsize]]
            y_batch = y_data[perm[i:i + batchsize]]

            if self.is_gpu >= 0:
                x_batch = chainer.cuda.to_gpu(x_batch)
                y_batch = chainer.cuda.to_gpu(y_batch)

            # 勾配の初期化
            self.optimizer.zero_grads()
            err, acc = self.validate(x_batch, y_batch, train=True)

            err.backward()
            self.optimizer.update()

            sum_error += float(chainer.cuda.to_cpu(err.data)) * batchsize
            sum_accuracy += float(chainer.cuda.to_cpu(acc.data)) * batchsize
            action()
        return sum_error / N, sum_accuracy / N

    def test(self, x_data, y_data, action=(lambda: None)):
        if self.is_gpu >= 0:
            x_data = chainer.cuda.to_gpu(x_data)
            y_data = chainer.cuda.to_gpu(y_data)

        err, acc = self.validate(x_data, y_data, train=False)
        action()

        return float(chainer.cuda.to_cpu(err.data)), float(chainer.cuda.to_cpu(acc.data))
