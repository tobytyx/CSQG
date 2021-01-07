# -*- coding: utf-8 -*-
import copy
import torch
from icecream import ic
from torch.utils.data import DataLoader
from data_provider.gw_dataset import GWDataset
from process_data.data_preprocess import get_games
from utils import constants
import numpy as np


class QuestionDataset(GWDataset):
    def __init__(self, data_dir, dataset, args, tokenizer):
        super(QuestionDataset,  self).__init__(data_dir, dataset, args, tokenizer)
        self.multi_cate = args["multi_cate"]

    def _load_data_from_dataset(self, data_dir, dataset, state_filter=("success",), multi_cate=False):

        old_games = get_games(data_dir, dataset, multi_cate)
        # 加上数据filter，根据状态
        old_games = [g for g in old_games if g.status in state_filter]
        # preprocess games, split a dialog into a few sentences
        self.games = []
        for g in old_games:
            for i in range(len(g.questions)):
                new_game = copy.copy(g)
                new_game.questions = []
                new_game.answers = []
                new_game.q_cates = []
                for j in range(0, i):
                    new_game.questions.append(g.questions[j])
                    new_game.answers.append(g.answers[j])
                    new_game.q_cates.append(g.q_cates[j])
                new_game.questions.append(g.questions[i])
                new_game.q_cates.append(g.q_cates[i])
                self.games.append(new_game)

    def __len__(self):
        return len(self.games)

    def __getitem__(self, item):
        game = self.games[item]
        questions = []
        answers = []
        categories = []
        q_indexs = []
        cur_index = -1
        game_id = game.id
        for i, (q, a, c) in enumerate(zip(game.questions, game.answers, game.q_cates)):
            # ic(i, q, a, c)
            q_tokens = self.tokenizer.apply(q)
            questions.extend(q_tokens)
            cur_index += len(q_tokens)
            q_indexs.append(cur_index)
            ans = constants.answers[a.lower()]
            answers.append(ans)
            if not self.multi_cate:
                c = c.index(1) if 1 in c else 3
            categories.append(c)
        game_turn = len(q_indexs)
        questions.append(constants.EOS)
        if game_turn == 0:  # no history
            answers.append(2)
            if self.multi_cate:
                categories.append([0, 0, 0, 0])
            else:
                categories.append(3)
            q_indexs.append(cur_index+1)
        if len(game.questions) - len(game.answers) == 1:
            if self.multi_cate:
                y_cate = game.q_cates[-1]
            else:
                y_cate = game.q_cates[-1].index(1) if 1 in game.q_cates[-1] else 3
            y = [constants.SOS] + self.tokenizer.apply(game.questions[-1]) + [constants.EOS]
        else:  # loop时的情况
            index = min(len(game.questions), len(game.gt_q_cates) - 1)
            if self.multi_cate:
                y_cate = game.gt_q_cates[index]
            else:
                y_cate = game.gt_q_cates[index].index(1) if 1 in game.gt_q_cates[index] else 3
            y = [constants.SOS] + [constants.EOS]
        if self.config["object"]:
            img = self.object_builder.load_feature(game.img.filename)
            bbox = img["pos"]
            img = img["att"]
            bbox = get_bbox(bbox)
        else:  # image
            img = self.image_builder.load_feature(game.img.id)
            bbox = np.random.randn(36, 76)
        q_len, turn = len(questions), len(q_indexs)
        return questions, q_len, q_indexs, answers, categories, turn, img, bbox, y_cate, y, game_id, game_turn


def question_collate(batch):
    questions, q_lens, q_indexs, answers, categories, turns, img, bbox, y_cates, ys, game_ids, game_turns = zip(*batch)
    is_multi_cate = isinstance(categories[0][0], list)
    max_qs_len = max(q_lens)
    max_turn = max(turns)
    y_lens = [len(y) for y in ys]
    max_y_len = max(y_lens)
    # 0
    pad_questions = [question + (max_qs_len - q_lens[i]) * [0] for i, question in enumerate(questions)]
    questions = torch.tensor(pad_questions, dtype=torch.long)
    # 1
    qs_lens = torch.tensor(q_lens, dtype=torch.int64)
    # 2
    pad_q_indexs = [q_index + (max_turn - turns[i]) * [0] for i, q_index in enumerate(q_indexs)]
    q_indexs = torch.tensor(pad_q_indexs, dtype=torch.int64)
    # 3
    pad_answers = [answer + (max_turn - turns[i]) * [2] for i, answer in enumerate(answers)]
    answers = torch.tensor(pad_answers, dtype=torch.long)
    # 4
    if is_multi_cate:
        pad_cates = [cate + (max_turn - turns[i]) * [[0] * constants.c_len] for i, cate in enumerate(categories)]
        categories = torch.tensor(pad_cates, dtype=torch.float)
    else:
        pad_cates = [cate + (max_turn - turns[i]) * [constants.c_len-1] for i, cate in enumerate(categories)]
        categories = torch.tensor(pad_cates, dtype=torch.long)
    # 6
    img = torch.tensor(img)
    # 7
    bbox = torch.tensor(bbox)
    # 8
    if is_multi_cate:
        y_cates = torch.tensor(y_cates, dtype=torch.float)
    else:
        y_cates = torch.tensor(y_cates, dtype=torch.long)
    # 9
    pad_ys = [y + (max_y_len - y_lens[i]) * [0] for i, y in enumerate(ys)]
    ys = torch.tensor(pad_ys, dtype=torch.long)
    # 5
    turns = torch.tensor(turns, dtype=torch.int64)
    return questions, qs_lens, q_indexs, answers, categories, turns, img, bbox, y_cates, ys, game_ids, game_turns


def get_weight(dataset):
    weights = []
    index2cate = {v: k for k, v in constants.categories.items()}
    for sample in dataset:
        cates = sample[-4]
        cates = [index2cate[i] for i in range(len(cates)) if cates[i] == 1]
        weight = sum([constants.category_weight[cate] for cate in cates]) / 10
        weights.append(weight)
    return weights


def get_bbox(bbox):
    """
    :param bbox: numpy, 36 * [x_min, y_min, x_max, y_max]
    :return: numpy, 36 * [x, y, w, h, relative_x0, ..., relative_x35, relative_y0, relative_y35](76)
    """
    mu = np.mean(bbox, axis=0)
    sigma = np.std(bbox, axis=0)
    bbox = (bbox-mu) / sigma
    w = bbox[:, 2] - bbox[:, 0]
    w = w.reshape(w.shape[0], 1)
    h = bbox[:, 3] - bbox[:, 1]
    h = h.reshape(h.shape[0], 1)
    pos_x = (bbox[:, 2] + bbox[:, 0]) / 2
    pos_y = (bbox[:, 3] + bbox[:, 1]) / 2
    rel_x = np.array([pos_x[i] - pos_x for i in range(bbox.shape[0])])
    rel_y = np.array([pos_y[i] - pos_y for i in range(bbox.shape[0])])
    pos_x = pos_x.reshape(pos_x.shape[0], 1)
    pos_y = pos_y.reshape(pos_y.shape[0], 1)
    return np.concatenate([pos_x, pos_y, w, h, rel_x, rel_y], axis=1)


def prepare_dataset(data_dir, mode, args, tokenizer):
    if mode == "train":
        train_set = QuestionDataset(data_dir, 'train', args, tokenizer=tokenizer)
        val_set = QuestionDataset(data_dir, 'valid', args, tokenizer=tokenizer)
        train_loader = DataLoader(train_set, batch_size=args["batch_size"], shuffle=True, num_workers=4,
                                  collate_fn=question_collate)
        val_loader = DataLoader(val_set, batch_size=args["batch_size"], shuffle=False, num_workers=4,
                                collate_fn=question_collate)
        return train_loader, val_loader
    else:
        test_set = QuestionDataset(data_dir, 'test', args, tokenizer=tokenizer)
        test_loader = DataLoader(test_set, batch_size=args["batch_size"], shuffle=False, num_workers=4,
                                 collate_fn=question_collate)
        return test_loader


def main():
    from arguments.qgen_args import qgen_arguments
    from process_data.tokenizer import GWTokenizer
    data_dir = "./../data/"
    tokenizer = GWTokenizer('./../data/dict.json')
    parser = qgen_arguments()
    args = parser.parse_args()
    args = vars(args)
    dataset = QuestionDataset(data_dir, 'test', args, tokenizer=tokenizer)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=8, collate_fn=question_collate)
    print(len(dataset), len(dataloader))
    dataiter = iter(dataloader)
    for i in range(1):
        batch = next(dataiter)


if __name__ == "__main__":
    main()
