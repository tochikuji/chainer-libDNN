# coding: utf-8

import chainer
import numpy
import matplotlib
import matplotlib.pyplot as plt
import cv2


class Visualizer(object):
    def __init__(self, network):
        self.nnbase = network
        self.model = network.model

        plt.subplots_adjust(hspace=0.5)

    def __convert_filters(self, layer, shape=(), T=False):
        layer = self.model[layer]
        self.bitmap = []
        weight = []
        if not T:
            weight = chainer.cuda.to_cpu(layer.W)
        else:
            weight = chainer.cuda.to_cpu(layer.W.T)

        if shape:
            for bitmap in weight:
                self.bitmap.append(bitmap.reshape(shape))
        else:
            for bitmap in weight:
                self.bitmap.append(bitmap[0])

    def plot_filters(self, layer, shape=(), T=False, title=True, interpolation=False):
        int_mode = 'none'
        if interpolation:
            int_mode = 'hermite'

        self.__convert_filters(layer, shape, T)
        N = len(self.bitmap)
        nrow = int(numpy.sqrt(N)) + 1

        for i in range(N):
            ax = plt.subplot(nrow, nrow, i + 1)
            if title:
                ax.set_title('filter %d' % (i + 1), fontsize=10)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            plt.imshow(self.bitmap[i], interpolation=int_mode, cmap=matplotlib.cm.gray)

        plt.show()

    def write_filters(self, layer, path='./', identifier='img', type='bmp', shape=(), T=False):
        self.__convert_filters(layer, shape, T)
        N = len(self.bitmap)
        # length of file indexes
        maxlen = int(numpy.log10(N)) + 1
        form = '{0:0>' + str(maxlen) + '}'

        fmax = numpy.max(self.bitmap)
        fmin = numpy.min(self.bitmap)

        self.bitmap = ((self.bitmap - fmin) * 0xff / (fmax - fmin)).astype(numpy.uint8)

        for i in range(N):
            filename = path + '/' + identifier + form.format(i) + '.' + type
            cv2.imwrite(filename, self.bitmap[i])

    def save_raw_filter(self, dst):
        for i in range(len(self.bitmap)):
            numpy.savetxt(dst + '/%d' % (i + 1) + '.csv', self.bitmap[i], delimiter=',')

    def __apply_filter(self, x, layer):
        output = self.nnbase.output(x, layer)

        # chainer.Variable -> numpy.ndarray (of GPUArray)
        return chainer.cuda.to_cpu(output).data

    def plot_output(self, x, layer):
        output = self.__apply_filter(x, layer)
        N = output.shape[0] * output.shape[1]
        nrow = int(numpy.sqrt(N)) + 1

        j = 0
        for batch in output:
            j += 1
            i = 0
            for img in batch:
                i += 1
                ax = plt.subplot(nrow, nrow, (j - 1) * output.shape[1] + i)
                ax.set_title('img%d-filt%d' % (j + 1, i + 1), fontsize=10)
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)

                plt.imshow(chainer.cuda.to_cpu(img), interpolation='none', cmap=matplotlib.cm.gray)

        plt.show()

    def write_output(self, x, layer, path='./', identifier='img_', type='bmp'):
        output = self.__apply_filter(x, layer)
        maxlen_t = int(numpy.log10(output.shape[0])) + 1
        tform = '{0:0>' + str(maxlen_t) + '}'
        maxlen_f = int(numpy.log10(output.shape[1])) + 1
        fform = '{0:0>' + str(maxlen_f) + '}'

        j = 0
        for batch in output:
            j += 1
            i = 0
            for img in batch:
                i += 1
                bitmap = chainer.cuda.to_cpu(img)
                fmax = numpy.max(bitmap)
                fmin = numpy.min(bitmap)

                bitmap = ((bitmap - fmin) * 0xff / (fmax - fmin)).astype(numpy.uint8)

                filename = path + '/' + identifier + tform.format(j) + '_f' + fform.format(i) + '.' + type
                cv2.imwrite(filename, bitmap)

    def write_activation(self, x, layer, path='./', identifier='img_', type='bmp'):
        output = self.__apply_filter(numpy.array([x]).astype(numpy.float32), layer)
        fform = '{0:0>' + str(int(numpy.log10(output.shape[1])) + 1) + '}'

        # filter num
        i = 0
        for img in output[0]:
            i += 1
            bitmap = chainer.cuda.to_cpu(img)
            fmax = numpy.max(bitmap)
            fmin = numpy.min(bitmap)

            bitmap = ((bitmap - fmin) * 0xff / (fmax - fmin)).astype(numpy.uint8)

            filename = path + '/' + identifier + 'f' + fform.format(i) + '.' + type
            cv2.imwrite(filename, bitmap)
