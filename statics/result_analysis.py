#-*- coding: utf-8 -*-
import json
import os
import time
import random
from collections import Counter
from process_data.data_preprocess import Game, get_games
from visualize import plot_game
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']  # 解决中文乱码
data_dir = "./../data/"
out_dir = "./../out/loop/"

class Analysis():
    def __init__(self, data_dir, filename):
        self.filename = os.path.join(data_dir, filename)
        self._load_games()

    def _load_json_file(self):
        games = []
        with open(self.filename, 'r') as f:
            for line in f:
                game = json.loads(line.strip('\n'))
                g = Game(**game)
                games.append(g)
        return games

    def _load_games(self):
        games = self._load_json_file()
        totalNum = len(games)
        succNum = 0
        failNum = 0
        self.games = games
        self.sgames = []
        self.fgames = []
        for game in self.games:
            if game.status == "success":
                succNum += 1
                self.sgames.append(game)
            else:
                failNum += 1
                self.fgames.append(game)
        print("There are total {0} games".format(totalNum))
        print("In it, {0} succeed, {1} fail, and the accuracy is {2}".format(succNum, failNum, succNum / totalNum))

    def globalAnaysis(self, games):
        qas_num = 0
        qas_length = 0
        for game in games:
            qas_num += len(game.questions)
            for q in game.questions:
                qas_length += len(q.split(" "))

        print("Average one game has {0} dialog turns.".format(qas_num/len(games)))
        print("Average one question has {0} words.".format(qas_length/qas_num))

    def repetitiveJudgment(self):
        # 对生成句子的重复性进行分析
        # diff_games = [self.games, self.sgames, self.fgames]
        diff_games = [self.games]
        games_dic = {0:"total games", 1:"successful games", 2:"failure games"}
        plt.figure(figsize=(6, 9))  # 调节图形大小
        for index in range(len(diff_games)):
            games = diff_games[index]
            rep_dialog = 0
            rep_num_dic = {0:0, 1:0, 2:0, 3:0, 4:0}
            for game in games:
                ques = game.questions
                rep_num = 0
                for i in range(1, len(ques)):
                    for j in range(i-1, -1, -1):
                        if ques[i] == ques[j]:
                            rep_num += 1
                            break
                rep_num_dic[rep_num] += 1
                if rep_num > 0:
                    rep_dialog += 1
            print(games_dic[index] + ":")
            print("{0} games has repetitive question".format(rep_dialog))
            print("{0} games repeat one question".format(rep_num_dic[1]))
            print("{0} games repeat two question".format(rep_num_dic[2]))
            print("{0} games repeat three question".format(rep_num_dic[3]))
            print("{0} games repeat four question".format(rep_num_dic[4]))
            print(rep_num_dic)
            # plt.subplot(1,3,index+1)
            # plt.pie(x=list(rep_num_dic.values()), labels=list(rep_num_dic.keys()),autopct='%3.2f%%')
            plt.bar(list(rep_num_dic.keys()), list(rep_num_dic.values()))

            # plt.legend()

            plt.xlabel("repetitive numbers")
            plt.ylabel("game numbers")

            plt.title(u"repetitive analysis")

            plt.show()

    def uselessAnalysis(self):
        # game中是否含有生成的无效句子
        diff_games = [self.games, self.sgames, self.fgames]
        games_dic = {0: "total games", 1: "successful games", 2: "failure games"}
        useless_questions = ["is it a ?", "is it the ?", "is it a a ?", "is it the the ?", "is it a the ?"]
        for index in range(len(diff_games)):
            games = diff_games[index]
            useless_dia_num = 0
            useless_q_num = 0
            q_useless = 0
            for game in games:
                ques = game.questions
                for i in range(1, len(ques)):
                    if ques[i] in useless_questions:
                        q_useless += 1
                useless_q_num += q_useless
                if q_useless:
                    useless_dia_num += 1

            print(games_dic[index] + ": has total {} games".format(len(diff_games[index])))
            print("{0} games has ussless question".format(useless_dia_num))
            if useless_dia_num:
                print("Average one dialog which has useless question has {} useless question.".
                      format(useless_q_num/useless_dia_num))

    def frequencyAnalysis(self, num=10):
        # game中生成的句子进行频率分析
        diff_games = [self.games, self.sgames, self.fgames]
        games_dic = {0: "total games", 1: "successful games", 2: "failure games"}
        for index in range(len(diff_games)):
            games = diff_games[index]
            question_counter = Counter()
            question_num = 0
            for game in games:
                ques = game.questions
                for q in ques:
                    question_counter[q] = question_counter[q] + 1
                    question_num += 1
            print(games_dic[index] + " {} most common question :".format(num))
            ques = question_counter.most_common(num)
            for i in range(num):
                print(ques[i][0] + "-> occur {} times".format(ques[i][1]))

    def random_visualize(self, num=5):
        diff_games = [self.sgames, self.fgames]
        for index in range(len(diff_games)):
            games = diff_games[index]
            for i in range(num):
                index = random.randint(0, len(games)-1)
                plot_game(games[index])


class CompareAna():
    def __init__(self, data_dir, filename1, filename2, filename3):
        self.filename1 = filename1
        self.filename2 = filename2
        self.filename3 = filename3
        self._load_games()

    def _load_json_file(self, filename):
        games = []
        with open(filename, 'r') as f:
            for line in f:
                game = json.loads(line.strip('\n'))
                g = Game(**game)
                games.append(g)
        return games

    def _load_games(self):
        games_1 = self._load_json_file(self.filename1)
        games_2 = self._load_json_file(self.filename2)
        games_3 = self._load_json_file(self.filename3)
        totalNum = len(games_1)
        print("There are total {0} games".format(totalNum))
        succNum = 0
        self.games_1 = games_1
        self.games_2 = games_2
        self.games_3 = games_3
        for game in self.games_1:
            if game.status == "success":
                succNum += 1
        print("In first file, {0} succeed, and the accuracy is {1}".format(succNum, succNum / totalNum))
        succNum = 0
        for game in self.games_2:
            if game.status == "success":
                succNum += 1
        print("In second file, {0} succeed, and the accuracy is {1}".format(succNum, succNum / totalNum))
        succNum = 0
        for game in self.games_3:
            if game.status == "success":
                succNum += 1
        print("In second file, {0} succeed, and the accuracy is {1}".format(succNum, succNum / totalNum))

    def random_visualize(self, num=5):
        for i in range(num):
            index = random.randint(0, len(self.games_1)-1)
            print(index)
            plot_game(self.games_1[index])
            time.sleep(5)
            plot_game(self.games_2[index])
            time.sleep(5)
            plot_game(self.games_3[index])

    def visualize(self, index):
        print(self.games_1[index].id)
        plot_game(self.games_1[index])
        time.sleep(5)
        plot_game(self.games_2[index])
        time.sleep(5)
        plot_game(self.games_3[index])

def single_ana():
    file = os.path.join(out_dir, "va_expand.json")
    ana = Analysis(data_dir, file)
    ana.globalAnaysis(ana.games)
    ana.globalAnaysis(ana.sgames)
    ana.globalAnaysis(ana.fgames)

    ana.repetitiveJudgment()
    ana.uselessAnalysis()
    ana.frequencyAnalysis()
    ana.random_visualize()

def compare_ana():
    file1 = os.path.join(out_dir, "model2(hred).json")
    file2 = os.path.join(out_dir, "model4.json")
    file3 = os.path.join(out_dir, "model1(enc-dec).json")
    # 1 \ 2 \ 4
    """
    ana1 = Analysis(data_dir, file3)
    ana2 = Analysis(data_dir, file1)
    ana3 = Analysis(data_dir, file2)
    ana1.repetitiveJudgment()
    ana2.repetitiveJudgment()
    ana3.repetitiveJudgment()
    """
    com = CompareAna(data_dir, file3, file1, file2)
    # com.random_visualize(1)
    com.visualize(9)

ana = Analysis(out_dir, "enc-dec-vgg.json")
ana.repetitiveJudgment()
ana = Analysis(out_dir, "hred-rcnn.json")
ana.repetitiveJudgment()
ana = Analysis(out_dir, "hred-va.json")
ana.repetitiveJudgment()