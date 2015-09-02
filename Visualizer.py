# coding: utf-8

import chainer
import numpy
import matplotlib
import matplotlib.pyplot as plt
import cv2


class Visualizer(object):
    def __init__(self, model, layer):
        self.model = model
        self.layer = model[layer]
        plt.subplots_adjust(hspace=0.5)

    def convert_filters(self, height=0, width=0):
        self.bitmap = []
        weight = chainer.cuda.to_cpu(self.layer.W)
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
