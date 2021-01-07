# -*- coding: utf-8 -*-
import copy
import torch
from torch.utils.data import DataLoader
from data_provider.gw_dataset import GWDataset
from process_data.data_preprocess import get_games
from utils import constants
import numpy as np


class QuestionDataset(GWDataset):
    def __init__(self, data_dir, dataset, config, tokenizer):
        super(QuestionDataset,  self).__init__(data_dir, dataset, config, tokenizer)

    def _load_data_from_dataset(self, data_dir, dataset, state_filter=("success",), multi_cate=False):
        old_games = get_games(data_dir, dataset)
        # 加上数据filter，根据状态
        old_games = [g for g in old_games if g.status in state_filter]
        # preprocess games, split a dialog into a few sentences
        self.games = []
        for g in old_games:
            for i in range(len(g.questions)):
                new_game = copy.copy(g)
                new_game.questions = []
                new_game.answers = []
                for j in range(0, i):
                    new_game.questions.append(g.questions[j])
                    new_game.answers.append(g.answers[j])
                new_game.questions.append(g.questions[i])
                self.games.append(new_game)

    def __len__(self):
        return len(self.games)

    def __getitem__(self, item):
        game = self.games[item]
        dialogue = []
        a_indexes = []
        game_id = game.id
        for i, (q, a) in enumerate(zip(game.questions, game.answers)):
            q_tokens = self.tokenizer.apply(q)
            a_tokens = self.tokenizer.apply(a, is_answer=True)
            dialogue.extend(q_tokens)
            dialogue.extend(a_tokens)
            a_indexes.append(len(a_indexes) + len(q_tokens))
        game_turn = len(a_indexes)
        if len(a_indexes) == 0:
            dialogue.append(0)
            a_indexes.append(0)

        if len(game.questions) - len(game.answers) == 1:
            y = [constants.SOS] + self.tokenizer.apply(game.questions[-1]) + [constants.EOS]
        else:  # generate时, q为start_token
            y = [constants.SOS] + [constants.EOS]
        if self.config["object"]:
            img = self.object_builder.load_feature(game.img.filename)
            bbox = img["pos"]
            img = img["att"]
            bbox = get_bbox(bbox)
        else:  # image
            img = self.image_builder.load_feature(game.img.id)
            bbox = np.random.randn(36, 76)
        return dialogue, len(dialogue), a_indexes, len(a_indexes), img, bbox, y, game_id, game_turn


def question_collate(batch):
    dialogues, dial_lens, a_indexes, turns, img, bbox, ys, game_ids, game_turns = zip(*batch)
    max_dial_len = max(dial_lens)
    max_turn = max(turns)
    y_lens = [len(y) for y in ys]
    max_y_len = max(y_lens)
    pad_dial = [dial + (max_dial_len - dial_lens[i]) * [0] for i, dial in enumerate(dialogues)]
    dialogues = torch.tensor(pad_dial, dtype=torch.long)
    dial_lens = torch.tensor(dial_lens, dtype=torch.int64)
    pad_a_indexes = [a_index + (max_turn - turns[i]) * [0] for i, a_index in enumerate(a_indexes)]
    a_indexes = torch.tensor(pad_a_indexes, dtype=torch.int64)
    img = torch.tensor(img)
    if bbox is not None:
        bbox = torch.tensor(bbox)
    pad_ys = [y + (max_y_len - y_lens[i]) * [0] for i, y in enumerate(ys)]
    ys = torch.tensor(pad_ys, dtype=torch.long)
    turns = torch.tensor(turns, dtype=torch.int64)
    return dialogues, dial_lens, a_indexes, turns, img, bbox, ys, game_ids, game_turns


def get_bbox(bbox):
    """
    :param bbox: numpy, 32 * [x_min, y_min, x_max, y_max]
    :return: numpy, 32 * [x, y, w, h, relative_x0, ..., relative_x35, relative_y0, relative_y35](76)
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
