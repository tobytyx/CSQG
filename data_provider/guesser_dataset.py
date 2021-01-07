# -*- coding: utf-8 -*-
from torch.utils.data import DataLoader
import torch
from data_provider.gw_dataset import GWDataset
from process_data.data_preprocess import get_games


class GuesserDataset(GWDataset):
    def __init__(self, data_dir, dataset, config, tokenizer):
        super(GuesserDataset, self).__init__(data_dir, dataset, config, tokenizer)

    def _load_data_from_dataset(self, data_dir, dataset, state_filter=("success",), multi_cate=False):
        games = get_games(data_dir, dataset)
        self.games = [g for g in games if g.status in state_filter]

    def __len__(self):
        return len(self.games)

    def __getitem__(self, item):
        game = self.games[item]
        dialog = []  # total dialog history
        for i in range(len(game.questions)):
            dialog.extend(self.tokenizer.apply(game.questions[i], is_answer=False))
            dialog.extend(self.tokenizer.apply(game.answers[i], is_answer=True))
        img = 0
        if self.config["image"]:
            img = self.image_builder.load_feature(game.img.id)
        cats = []
        spas = []
        target = 0
        for i, o in enumerate(game.objects):
            cats.append(o.category_id)
            # N x [x_min, y_min, x_max, y_max, x_center, y_center, w_box, h_box]
            spas.append(o.spatial)
            if o.id == game.object_id:
                target = i  # 目标object的index
        return dialog, len(dialog), img, len(cats), cats, spas, target


def guesser_collate(batch):
    dialogs, dial_lens, imgs, obj_lens, cats, spas, ys = zip(*batch)
    max_dial_len = max(dial_lens)
    pad_dialogues = [dialogue + (max_dial_len - dial_lens[i]) * [0] for i, dialogue in enumerate(dialogs)]
    pad_dialogues = torch.tensor(pad_dialogues, dtype=torch.long)
    dial_lens = torch.tensor(dial_lens, dtype=torch.long)
    imgs = torch.tensor(imgs)
    max_obj_lens = max(obj_lens)
    pad_categories = [cat + (max_obj_lens - obj_lens[i]) * [0] for i, cat in enumerate(cats)]
    pad_categories = torch.tensor(pad_categories, dtype=torch.long)
    pad_spatials = [spa + (max_obj_lens - obj_lens[i]) * [[0]*8] for i, spa in enumerate(spas)]
    pad_spatials = torch.tensor(pad_spatials, dtype=torch.float)
    obj_mask = [[1] * obj_len + (max_obj_lens - obj_len) * [0] for obj_len in obj_lens]
    obj_mask = torch.tensor(obj_mask, dtype=torch.long)
    ys = torch.tensor(ys, dtype=torch.long)
    return pad_dialogues, dial_lens, imgs, obj_mask, pad_categories, pad_spatials, ys


def prepare_dataset(data_dir, mode, args, tokenizer):
    if mode == "train":
        train_set = GuesserDataset(data_dir, 'train', args, tokenizer=tokenizer)
        val_set = GuesserDataset(data_dir, 'valid', args, tokenizer=tokenizer)
        train_loader = DataLoader(train_set, batch_size=args["batch_size"], shuffle=True, num_workers=4,
                                  collate_fn=guesser_collate)
        val_loader = DataLoader(val_set, batch_size=args["batch_size"], shuffle=False, num_workers=4,
                                collate_fn=guesser_collate)
        return train_loader, val_loader
    else:
        test_set = GuesserDataset(data_dir, 'test', args, tokenizer=tokenizer)
        test_loader = DataLoader(test_set, batch_size=args["batch_size"], shuffle=False, num_workers=4,
                                 collate_fn=guesser_collate)
        return test_loader
