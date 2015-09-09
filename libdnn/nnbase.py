# coding: utf-8
# Simple chainer interfaces for Deep learning researching
# Neural networks base template
# Author: Aiga SUZUKI <ai-suzuki@aist.go.jp>

import chainer
import numpy
import os.path
from abc import abstractmethod
from types import MethodType


class NNBase(object):
    def __init__(self, model, gpu=-1):
        self.model = model
        self.gpu = gpu

        if self.gpu >= 0:
            chainer.cuda.init(self.gpu)
            self.model = self.model.to_gpu()

    @abstractmethod
    def __forward(self, x, train):
        pass

    def forward(self, x_data, train=False):
        if self.gpu >= 0:
            x_data = chainer.cuda.to_gpu(x_data)

        x = chainer.Variable(x_data)

        return self.__forward(x, train)

    @abstractmethod
    def __output(self, x, layer):
        pass

    def output(self, x_data):
        if self.gpu >= 0:
            x_data = chainer.cuda.to_gpu(x_data)

        x = chainer.Variable(x_data)

        return self.__output(x)

    def set_forward(self, func):
        self.forward = MethodType(func, self, )

    def set_output(self, func):
        self.output = MethodType(func, self, NNBase)

    def set_loss_function(self, func, param={}):
        self.loss_function = func
        self.loss_param = param

    def set_optimizer(self, func, param={}):
        self.optimizer = func
        self.opt_param = param

    # save trained network parameters to file
    def save_param(self, dst='./network.param.npy'):
        param = numpy.array(self.model.to_cpu().parameters)
        numpy.save(dst, param)

    # load pre-trained network parameters from file
    def load_param(self, src='./network.param.npy'):
        if not os.path.exists(src):
            raise IOError('specified parameter file does not exists')

        param = numpy.load(src)
        self.model.copy_parameters_from(param)

        # by this process, model parameters is cpu_array now
        if self.gpu >= 0:
            self.model = self.model.to_gpu()
