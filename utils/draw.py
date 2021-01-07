import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg  # mpimg 用于读取图片
from process_data.data_preprocess import get_games

def mp_show(filename):
    # use matplotlib to show image
    lena = mpimg.imread(filename)  # 读取和代码处于同一目录下的 lena.png
    # 此时 lena 就已经是一个 np.array 了，可以对它进行任意处理
    plt.imshow(lena)  # 显示图片
    plt.axis('off')  # 不显示坐标轴
    plt.show()


class AbstractPlotter(object):
    def __init__(self, path, name, suffix):
        self.path = path
        self.name = suffix + "." + name

    def save_as_pdf(self):
        from matplotlib.backends.backend_pdf import PdfPages
        with PdfPages(os.path.join(self.path, self.name+str(".pdf"))) as pdf:
            pdf.savefig()
            plt.close()

    def plot(self):
        plt.plot()


class DialogueSampler(AbstractPlotter):
    def __init__(self, path, games, logger, suffix):
        super(DialogueSampler, self).__init__(path, self.__class__.__name__, suffix)

        for i, game in enumerate(games[:100]):

            logger.info("Dialogue {} : ".format(i))
            logger.info(" - picture : http://mscoco.org/images/{}".format(game.image.id))
            logger.info(" - object category : {}".format(game.object.category))
            logger.info(" - object position : {}, {}".format(game.object.bbox.x_center,game.object.bbox.y_center))
            logger.info(" - question/answers :")
            for q, a in zip(game.questions, game.answers):
                logger.info('  > ' + q + ' ? ' + a)
            logger.info('  #> ' + game.status)
            logger.info("")


    def save_as_pdf(self):
        pass


    def plot(self):
        pass
