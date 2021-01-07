# -*- coding: utf-8 -*-
import copy
from torch.utils.data import DataLoader
import torch
from data_provider.gw_dataset import GWDataset
from process_data.data_preprocess import get_games


class OracleDataset(GWDataset):
    def __init__(self, data_dir, dataset, config, tokenizer=None):
        super(OracleDataset, self).__init__(data_dir, dataset, config, tokenizer)

    def _load_data_from_dataset(self, data_dir, dataset, state_filter=("success",), multi_cate=False):
        # train/valid/test load dataset
        answers = {'yes': 0, 'no': 1, 'n/a': 2}
        old_games = get_games(data_dir, dataset)
        self.games = []
        for game in old_games:
            for i, q, a in zip(game.question_ids, game.questions, game.answers):
                new_game = copy.copy(game)
                new_game.questions = q
                new_game.answers = answers[a.lower()]
                new_game.question_ids = i
                self.games.append(new_game)

    def __len__(self):
        return len(self.games)

    def __getitem__(self, item):
        game = self.games[item]
        img, crop = 0, 0
        if self.config["image"]:
            img = self.image_builder.load_feature(game.img.id)
        if self.config["crop"]:
            crop = self.crop_builder.load_feature(game.object.id)
        category = game.object.category_id
        spatial = game.object.spatial
        question = self.tokenizer.apply(game.questions, is_answer=False)
        y = game.answers if game.answers is not None else 0
        return question, len(question), img, crop, category, spatial, y


def oracle_collate(batch):
    questions, q_lens, imgs, crops, cats, spas, ys = zip(*batch)
    max_q_len = max(q_lens)
    pad_questions = [q + (max_q_len - q_lens[i]) * [0] for i, q in enumerate(questions)]
    pad_questions = torch.tensor(pad_questions, dtype=torch.long)
    q_lens = torch.tensor(q_lens, dtype=torch.long)
    imgs = torch.tensor(imgs)
    crops = torch.tensor(crops, dtype=torch.float)
    cats = torch.tensor(cats, dtype=torch.long)
    spas = torch.tensor(spas, dtype=torch.float)
    ys = torch.tensor(ys, dtype=torch.long)
    return pad_questions, q_lens, imgs, crops, cats, spas, ys


def prepare_dataset(data_dir, mode, flags, tokenizer):
    if mode == "train":
        train_set = OracleDataset(data_dir, 'train', flags, tokenizer=tokenizer)
        val_set = OracleDataset(data_dir, 'valid', flags, tokenizer=tokenizer)
        train_loader = DataLoader(train_set, batch_size=flags["batch_size"], shuffle=True, num_workers=8,
                                  collate_fn=oracle_collate)
        val_loader = DataLoader(val_set, batch_size=flags["batch_size"], shuffle=False, num_workers=8,
                                collate_fn=oracle_collate)
        return train_loader, val_loader
    else:
        test_set = OracleDataset(data_dir, 'test', flags, tokenizer=tokenizer)
        test_loader = DataLoader(test_set, batch_size=flags["batch_size"], shuffle=False, num_workers=8,
                                 collate_fn=oracle_collate)
        return test_loader
