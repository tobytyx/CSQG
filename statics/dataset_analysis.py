from process_data.data_preprocess import get_games
import collections
import matplotlib.pyplot as plt
import numpy as np

data_dir = "./../../data/"

class DataAnalysis():
    def __init__(self, data_dir="../data/"):
        self.data_dir = data_dir
        self.games = []
        self.train_games = get_games(data_dir, 'train')
        self.valid_games = get_games(data_dir, 'valid')
        self.test_games = get_games(data_dir, 'test')
        self.games.extend(self.train_games)
        self.games.extend(self.valid_games)
        self.games.extend(self.test_games)
        self.game_number_analysis()
        self.image_and_object_analysis()
        self.status_anlysis()
        self.object_category_analyse()
        self.answer_analysis()
        self.setence_length_anlysis()
        self.turn_analysis()
        self.plot_bar_graph(self.sentence_count, "Length analysis of dataset's question", "Question length",
                            "Ratio of dialogs",  "length_analysis.jpg")
        self.plot_bar_graph(self.turn_count, "Turn analysis of  dataset's dialogs", "Number of question",
                            "Ratio of dialogs", "turn_analysis.jpg")
        self.plt_graph()


    def game_number_analysis(self):
        print("total games: {}".format(len(self.games)))
        print("train_games: {}".format(len(self.train_games)))
        print("valid_games: {}".format(len(self.valid_games)))
        print("test_games: {}".format(len(self.test_games)))

    def image_and_object_analysis(self):
        image = []
        object = []
        for game in self.games:
            image.append(game.img.filename)
            for o in game.objects:
                object.append(o.id)
        image_count = collections.Counter(image)
        object_count = collections.Counter(object)
        print("total image:{}".format(len(image_count.items())))
        print("total object:{}".format(len(object_count.items())))

    def status_anlysis(self):
        status = []
        for game in self.games:
            status.append(game.status)
        status_count = collections.Counter(status)
        for key, value in status_count.items():
            print(str(key) + "->" + str(value))

    def object_category_analyse(self):
        # 物体类别分析
        object_cate = []
        for game in self.games:
            for o in game.objects:
                category = o.category_id
                object_cate.append(category)

        count = collections.Counter(object_cate)
        print(len(count.items()))
        print(max(count.keys()))
        for key,value in count.items():
            print(str(key) + "->" + str(value))

    def answer_analysis(self):
        # 对回答进行分析
        all_answer = []
        for game in self.games:
            all_answer.extend(game.answers)
        count = collections.Counter(all_answer)
        for key,value in count.items():
            print(str(key) + "->" + str(value/len(all_answer)))

    def setence_length_anlysis(self):
        sentence_length = []
        qas_num = 0
        for game in self.games:
            for q in game.questions:
                q = q.split(" ")
                sentence_length.append(len(q))
                qas_num += 1
        self.sentence_count = collections.Counter(sentence_length)
        for key,value in self.sentence_count.items():
            print(str(key) + "->" + str(value))
        print("dataset's average question length is {}".format(sum(sentence_length)/len(sentence_length)))
        print("dataset's total qa's number is {}".format(qas_num))

    def turn_analysis(self):
        turns = []
        for game in self.games:
            turns.append(len(game.questions))
        print("one dialog has avarage {} qas".format(sum(turns)/len(turns)))
        self.turn_count = collections.Counter(turns)
        for key,value in self.turn_count.items():
            print(str(key) + "->" + str(value))

    def plot_bar_graph(self, counter, title, xlabel, ylabel, filename):
        x = list(counter.keys())
        total = sum(counter.values())
        y = [v/total for v in counter.values()]
        plt.bar(x,y)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.savefig(filename)
        plt.show()

    def plt_graph(self):
        plt.figure(figsize=(12, 6))
        plt.subplot(1,2,1)
        x = list(self.turn_count.keys())
        total = sum(self.turn_count.values())
        y = [v / total for v in self.turn_count.values()]
        plt.bar(x, y)
        plt.xlabel("Number of question")
        plt.ylabel("Ratio of dialogs")
        plt.subplot(1,2,2)
        x = list(self.sentence_count.keys())
        total = sum(self.sentence_count.values())
        y = [v / total for v in self.sentence_count.values()]
        plt.bar(x, y)
        plt.xlabel("Question length")
        plt.savefig("analysis of  dataset")


data = DataAnalysis()
