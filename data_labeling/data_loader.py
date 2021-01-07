# -*- coding: utf-8 -*-
import torch
from pytorch_transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader
from utils.constants import categories, BERT_BASE_UNCASED_PATH


def collate_fn(data):
    """
    :param data:
    :return:
    """
    ids, xs, labels = zip(*data)
    max_len = max(len(x) for x in xs)
    x_batch = torch.tensor([x + [0] * (max_len - len(x)) for x in xs], dtype=torch.long)
    x_mask = torch.tensor([[1] * len(x) + [0] * (max_len - len(x)) for x in xs], dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.long)
    return ids, x_batch, x_mask, labels


class BertDataset(Dataset):
    def __init__(self, data, mode, multi=False):
        self.ids, self.texts, self.labels = [], [], []
        self.multi = multi
        for data in data:
            self.ids.append(data[0])
            self.texts.append(data[1])
            l_list = [0, 0, 0, 0] if self.multi else 0
            if mode != "classify":
                if self.multi:
                    for label in data[2]:
                        l_list[categories[label]] = 1
                else:
                    l_list = categories[data[2][0]]
            self.labels.append(l_list)
        self.tokenizer = BertTokenizer.from_pretrained(BERT_BASE_UNCASED_PATH)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        tid, text, label = self.ids[idx], self.texts[idx], self.labels[idx]
        # ic(text, title, topic)
        text = self.tokenizer.tokenize(text)
        x = ["[CLS]"] + text + ["[SEP]"]
        x = self.tokenizer.convert_tokens_to_ids(x)
        return tid, x, label


def prepare_loader(data, args, mode="multi"):
    multi = True if mode == "multi" else False
    if args["option"] == "train":
        len_train = int(len(data) * 0.5)
        len_val = int(len(data) * 0.1)
        train_data = data[:len_train]
        val_data = data[len_train:len_train+len_val]
        train_loader = DataLoader(
            BertDataset(train_data, "train", multi), batch_size=args["batch_size"], collate_fn=collate_fn)
        val_loader = DataLoader(
            BertDataset(val_data, "train", multi), batch_size=args["batch_size"], collate_fn=collate_fn)
        return train_loader, val_loader
    elif args["option"] in "test":
        len_test = int(len(data) * 0.4)
        test_data = data[-len_test:]
        test_loader = DataLoader(BertDataset(test_data, "test", multi),
                                 batch_size=args["batch_size"], collate_fn=collate_fn)
        return test_loader
    else:
        data_loader = DataLoader(BertDataset(data, "classify", multi),
                                 batch_size=args["batch_size"], collate_fn=collate_fn)
        return data_loader
