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
            # if using pyCUDA version (v1.2.0 earlier)
            if chainer.__version__ <= '1.2.0':
                chainer.cuda.init(self.gpu)
            # CuPy (1.3.0 later) version
            else:
                chainer.cuda.get_device(self.gpu).use()

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

    def output(self, x_data, layer):
        if self.gpu >= 0:
            x_data = chainer.cuda.to_gpu(x_data)

        x = chainer.Variable(x_data)

        act = self.__output(x, layer)
        if act is None:
            raise ValueError("Network has returned strange value, invalid layer may be specified")

        return act

    def set_forward(self, func):
        self.__forward = MethodType(func, self, )

    def set_output(self, func):
        self.__output = MethodType(func, self, NNBase)

    def set_loss_function(self, func, param={}):
        self.loss_function = func
        self.loss_param = param

    def set_optimizer(self, func, param={}):
        self.optimizer = func(**param)
        self.optimizer.setup(self.model)

    # save trained network parameters to file
    def save_param(self, dst='./network.param.npy'):
        # model.to_cpu() seems to change itself
        # This causes step-by-step saving each epochs with gpu
        param = numpy.array(self.model.to_cpu().parameters)
        numpy.save(dst, param)
        if self.gpu >= 0:
            self.model.to_gpu()

    # load pre-trained network parameters from file
    def load_param(self, src='./network.param.npy'):
        if not os.path.exists(src):
            raise IOError('specified parameter file does not exists')

        param = numpy.load(src)
        self.model.copy_parameters_from(param)

        # by this process, model parameters to be cpu_array
        if self.gpu >= 0:
            self.model = self.model.to_gpu()
