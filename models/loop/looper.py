# -*- coding:utf-8 -*-
import json
import os
import random
from process_data.data_preprocess import get_games


class Looper(object):
    def __init__(self, data_dir, oracle, guesser, question, args):
        self.data_dir = data_dir
        self.oracle = oracle
        self.guesser = guesser
        self.question = question
        self.args = args

    def clear_data_for_new_pictures(self):
        multi_cate = self.question.config.get("multi_cate", False)
        game_data = get_games(self.data_dir, "test", multi_cate)
        for i in range(len(game_data)):
            game_data[i].questions = []
            game_data[i].answers = []
            game_data[i].q_cates = []
            game_data[i].status = "incomplete"
        return game_data

    def clear_data_for_new_objects(self):
        game_data = get_games(self.data_dir, "train")
        for i in range(len(game_data)):
            game_data[i].questions = []
            game_data[i].answers = []
            game_data[i].q_cates = []
            game_data[i].status = "incomplete"
            # random choose a object
            total_object = len(game_data[i].objects)
            choose_one = random.randint(0, total_object-1)  # this function including both end points.
            game_data[i].object_id = game_data[i].objects[choose_one].id
        return game_data

    @staticmethod
    def clear_data(game_data):
        for i in range(len(game_data)):
            game_data[i].questions = []
            game_data[i].answers = []
            game_data[i].q_cates = []
            game_data[i].status = "incomplete"
        return game_data

    def loop(self, game_data, data_analysis, visualize=False):
        # input new game_data(no history), iterate make use of three modules wrapper to generate a complete dialog
        # and finish  the game
        tmp_data = []
        for i in range(len(game_data)):
            tmp_data.append([game_data[i].id])
        for turn in range(self.args["max_turn"]):
            self.question.question(game_data, turn, visualize)
            self.oracle.answer(game_data)
        # let guesser to guess
        batch, success = self.guesser.guess(game_data)
        success_game_num = sum(success)
        for i in range(len(game_data)):
            tmp_data[i].append(game_data[i].status)
        data_analysis.extend(tmp_data)
        return batch, success_game_num

    def eval(self, option, out_dir, store=False, visualize=False):
        # for given dataset to eval dialog accuracy
        if option == "new_pictures":
            game_data = self.clear_data_for_new_pictures()
        else:
            game_data = self.clear_data_for_new_objects()
        bsz = self.args["batch_size"]
        batch_num = len(game_data) // bsz
        total = 0
        success = 0
        data_analysis = []
        print("total data size: {0}".format(len(game_data)))
        print("there are {0} batches".format(batch_num))
        for i in range(batch_num):
            # get batch data
            batch = [game_data[j] for j in range(i*bsz, (i+1)*bsz) if j < len(game_data)]
            # play the game and generate full dialog
            size, success_game = self.loop(batch, data_analysis, visualize)
            total += size
            success += success_game
            if i % 100 == 0:
                print("batch: {} success rate: {:.4f}".format(i, success / total))
        success_rate = success / total
        print("{0} success rate is {1:.2f}%".format(option, success_rate*100))
        if store:
            self.store_game_data(game_data, out_dir)
        return game_data, success_rate

    def store_game_data(self, game_data, out_dir):
        datas = []
        for i in range(len(game_data)):
            data = {
                "id": game_data[i].id,
                "status": game_data[i].status,
                "qas": [{"answer": a, "question": q} for q, a in zip(
                    game_data[i].questions, game_data[i].answers)],
                "image": game_data[i].img.filename,
                "object_id": game_data[i].object_id,
                "objects": [o.json() for o in game_data[i].objects],
                "guess_id": game_data[i].guess_id if hasattr(game_data[i], "guess_id") else 0,
                "object": game_data[i].object.json(),
                "q_cates": game_data[i].q_cates
            }
            datas.append(data)
        turn = "{}t_".format(self.args["max_turn"])
        game_file = os.path.join(out_dir, "games", turn + self.args["qgen_name"] + ".json")
        with open(game_file, mode="w") as f:
            json.dump(datas, f, ensure_ascii=False)
