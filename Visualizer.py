# coding: utf-8

import chainer
import numpy
import matplotlib
import matplotlib.pyplot as plt
import cv2


class Visualizer(object):
    def __init__(self, model):
        self.nnbase = model
        self.model = model.model

        plt.subplots_adjust(hspace=0.5)

    def convert_filters(self, layer, height=0, width=0):
        layer = self.model[layer]
        self.bitmap = []
        weight = chainer.cuda.to_cpu(layer.W)
        for bitmap in weight:
            if height or width:
                self.bitmap.append(bitmap[0].reshape(height, width))
            else:
                self.bitmap.append(bitmap[0])

    def plot_filters(self):
        N = len(self.bitmap)
        nrow = int(numpy.sqrt(N)) + 1

        for i in range(N):
            ax = plt.subplot(nrow, nrow, i + 1)
            ax.set_title('filter %d' % (i + 1), fontsize=10)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            plt.imshow(self.bitmap[i], interpolation='none', cmap=matplotlib.cm.gray)

        plt.show()

    def write_filters(self, path='./', identifier='img', type='bmp'):
        N = len(self.bitmap)
        # 指定する最大のファイルインデックスサイズ
        maxlen = int(numpy.log10(N)) + 1
        form = '{0:0>' + str(maxlen) + '}'

        fmax = numpy.max(self.bitmap)
        fmin = numpy.min(self.bitmap)

        self.bitmap = ((self.bitmap - fmin) * 0xff / (fmax - fmin)).astype(numpy.uint8)

        for i in range(N):
            filename = path + '/' + identifier + form.format(i) + '.' + type
            cv2.imwrite(filename, self.bitmap[i])

    def __apply_filter(self, x, layer):
        if self.nnbase.is_gpu >= 0:
            x = chainer.cuda.to_gpu(x)

        x = chainer.Variable(x, volatile=True)

        output = self.nnbase.output(x, layer)
        if output is None:
            raise Exception("invalid layer was specified.")

        # chainer.Variable -> numpy.ndarray (of GPUArray)
        return chainer.cuda.to_cpu(output).data

    def plot_output(self, x, layer):
        output = self.__apply_filter(x, layer)
        print(output.shape)
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
