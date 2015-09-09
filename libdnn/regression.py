# coding: utf-8
# Simple chainer interfaces for Deep learning researching
# Author: Aiga SUZUKI <ai-suzuki@aist.go.jp>

import chainer
import numpy
import chainer.functions as F
import chainer.optimizers as Opt
from nnbase import NNBase


class Regressor(NNBase):
    def __init__(self, model, gpu=-1):
        NNBase.__init__(self, model, gpu)

        self.optimizer = Opt.Adam
        self.opt_param = {}

        self.loss_function = F.softmax_cross_entropy
        self.loss_param = {}

    def validate(self, x_data, t_data, train=False):
        y = self.forward(x_data, train=train)

        if self.gpu >= 0:
            t_data = chainer.cuda.to_gpu(t_data)

        t = chainer.variable(t_data)

        return self.loss_function(y, t, **self.loss_param), F.accuracy(y, t)

    def train(self, x_data, t_data, batchsize=100, action=(lambda: None)):
        # num of x_data
        N = len(x_data)
        perm = numpy.random.permutation(N)

        sum_accuracy = 0.
        sum_error = 0.

        for i in range(0, N, batchsize):
            # training mini batch of x, y
            x_batch = x_data[perm[i:i + batchsize]]
            t_batch = t_data[perm[i:i + batchsize]]

            # initialize optimizer gradients
            self.optimizer.zero_grads()
            err, acc = self.validate(x_batch, t_batch, train=True)

            err.backward()
            self.optimizer.update()

            sum_error += float(chainer.cuda.to_cpu(err.data)) * batchsize
            sum_accuracy += float(chainer.cuda.to_cpu(acc.data)) * batchsize
            action()

        return sum_error / N, sum_accuracy / N

    def test(self, x_data, t_data, action=(lambda: None)):
        err, acc = self.validate(x_data, t_data, train=False)
        action()

        return float(chainer.cuda.to_cpu(err.data)), float(chainer.cuda.to_cpu(acc.data))
