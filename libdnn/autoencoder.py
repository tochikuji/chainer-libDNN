# coding: utf-8
# Simple chainer interfaces for Deep learning researching
# For autoencoder
# Author: Aiga SUZUKI <ai-suzuki@aist.go.jp>

import chainer
import chainer.functions as F
import chainer.optimizers as Opt
import numpy
from libdnn.nnbase import NNBase
from types import MethodType
from abc import abstractmethod


class AutoEncoder(NNBase):
    def __init__(self, model, gpu=-1):
        NNBase.__init__(self, model, gpu)

        self.optimizer = Opt.Adam()
        self.optimizer.setup(self.model)

        self.loss_function = F.mean_squared_error
        self.loss_param = {}

    def validate(self, x_data, train=False):
        y = self.forward(x_data, train=train)
        if self.gpu >= 0:
            x_data = chainer.cuda.to_gpu(x_data)

        x = chainer.Variable(x_data)

        return self.loss_function(x, y, **self.loss_param)

    def train(self, x_data, batchsize=100, action=(lambda: None)):
        N = len(x_data)
        perm = numpy.random.permutation(N)

        sum_error = 0.

        for i in range(0, N, batchsize):
            x_batch = x_data[perm[i:i + batchsize]]

            self.optimizer.zero_grads()
            err = self.validate(x_batch, train=True)

            err.backward()
            self.optimizer.update()

            sum_error += float(chainer.cuda.to_cpu(err.data)) * len(x_batch)
            action()

        return sum_error / N

    def test(self, x_data, batchsize=100, action=(lambda: None)):
        N = len(x_data)
        perm = numpy.random.permutation(N)

        sum_error = 0.

        for i in range(0, N, batchsize):
            x_batch = x_data[perm[i:i + batchsize]]

            err = self.validate(x_batch, train=False)

            sum_error += float(chainer.cuda.to_cpu(err.data)) * batchsize
            action()

        return sum_error / N


class StackedAutoEncoder(AutoEncoder):
    def __init__(self, model, gpu=-1):
        self.sublayer = []
        AutoEncoder.__init__(self, model, gpu)

    def set_order(self, encl, decl):
        if len(encl) != len(decl):
            raise TypeError('Encode/Decode layers mismatch')

        self.depth = len(encl)

        for (el, dl) in zip(encl, reversed(decl)):
            self.sublayer.append(chainer.FunctionSet(
                enc=self.model[el],
                dec=self.model[dl]
            ))

    @abstractmethod
    def __encode(self, x, layer, train):
        pass

    def set_encode(self, func):
        self.__encode = MethodType(func, self, StackedAutoEncoder)

    def encode(self, x_data, layer=None, train=False):
        if self.gpu >= 0:
            x_data = chainer.cuda.to_gpu(x_data)

        x = chainer.Variable(x_data)

        return self.__encode(x, layer, train)

    @abstractmethod
    def __decode(self, x, layer, train):
        pass

    def set_decode(self, func):
        self.__decode = MethodType(func, self, StackedAutoEncoder)

    def decode(self, x_data, layer=None, train=False):
        if self.gpu >= 0:
            x_data = chainer.cuda.to_gpu(x_data)

        x = chainer.Variable(x_data)

        return self.__decode(x, layer, train)

    def forward(self, x_data, train=False):
        code = self.encode(x_data, train=train)
        y = self.__decode(code, train=train)

        return y

    def validate(self, x_data, layer=None, train=False):
        targ = self.encode(x_data, layer - 1, train=False)
        code = self.encode(x_data, layer, train=train)

        y = self.__decode(code, layer, train=train)

        return self.loss_function(targ, y, **self.loss_param)

    def train(self, x_data, batchsize=100, action=(lambda: None)):
        errs = []
        N = len(x_data)
        perm = numpy.random.permutation(N)

        for l in range(1, self.depth + 1):
            self.optimizer.setup(self.sublayer[l - 1])

            sum_error = 0.

            for i in range(0, N, batchsize):
                x_batch = x_data[perm[i:i + batchsize]]

                self.optimizer.zero_grads()
                err = self.validate(x_batch, layer=l, train=True)

                err.backward()
                self.optimizer.update()

                sum_error += float(chainer.cuda.to_cpu(err.data)) * len(x_batch)
                action()

            errs.append(sum_error / N)

        return tuple(errs)

    def test(self, x_data, batchsize=100, action=(lambda: None)):
        N = len(x_data)
        perm = numpy.random.permutation(N)

        sum_error = 0.

        for i in range(0, N, batchsize):
            x_batch = x_data[perm[i:i + batchsize]]
            y = self.forward(x_batch, train=False)

            if self.gpu >= 0:
                x_batch = chainer.cuda.to_gpu(x_batch)
            x = chainer.Variable(x_batch)

            err = self.loss_function(x, y, **self.loss_param)

            sum_error += float(chainer.cuda.to_cpu(err.data)) * len(x_batch)
            action()

        return sum_error / N
