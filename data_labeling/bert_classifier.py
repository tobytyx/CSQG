# -*- coding: utf-8 -*-
import argparse
import torch
import torch.nn as nn
from icecream import ic
from pytorch_transformers import AdamW, WarmupLinearSchedule, BertForSequenceClassification
from data_labeling.classifier import BertClassify
from data_labeling.data_loader import prepare_loader
import json
import os
import numpy as np
from utils import constants


parser = argparse.ArgumentParser()
parser.add_argument('--option', type=str, default="train", choices=['train', 'test', 'classify'])
parser.add_argument('--epoch', type=int, default=50)
parser.add_argument('--model', type=str, default="bert_cnn", choices=["bert", "bert_cnn"])
parser.add_argument("--name", default="bert_cnn", type=str)
parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--th', type=float, default=0.4)
parser.add_argument('--dropout', type=float, default=0.2)
parser.add_argument('--num_filters', type=int, default=8)
args = parser.parse_args()

args = vars(args)
checkpoint_dir = os.path.join("../out", args["name"])
if args["option"] in ["classify", "test"]:
    with open(os.path.join(checkpoint_dir, "args.json")) as f:
        new_args = json.load(f)
        new_args["option"] = args["option"]
    args = new_args
ic(args)

device = torch.device('cuda')
with open("data_labeling/data.json") as f:
    data = json.load(f)
if args["model"] == "bert":
    model = BertForSequenceClassification.from_pretrained(constants.BERT_BASE_UNCASED_PATH, num_labels=4)
else:
    model = BertClassify(args)
if args["option"] == "classify":
    model.load_state_dict(torch.load(os.path.join(checkpoint_dir, "model.bin")))
    data_loader = prepare_loader(data["total"], args, mode="single")
    model = model.to(device)
    i2c = {v: k for k, v in constants.categories.items()}
    id2cate = {}
    print("total step: {}".format(len(data_loader)))
    for _, batch in enumerate(data_loader):
        ids, x, mask, _ = batch
        x, mask = x.to(device), mask.to(device)
        outputs = model(x, attention_mask=mask)
        if args["model"] == "bert":
            outputs = outputs[0]
        labels = outputs.cpu().detach()
        _, labels = torch.max(labels, dim=1)
        batch_size = len(ids)
        for i in range(batch_size):
            id2cate[ids[i]] = [i2c[labels[i].item()]]
    with open(os.path.join(checkpoint_dir, "output.json"), mode="w") as f:
        json.dump(id2cate, f)
elif args["option"] == "test":
    model.load_state_dict(torch.load(os.path.join(checkpoint_dir, "model.bin")))
    model = model.to(device)
    test_loader = prepare_loader(data["train"], args, mode="single")
    model.eval()
    id2cate = {}
    i2c = {v: k for k, v in constants.categories.items()}
    confusion_matrix = np.zeros([constants.c_len, constants.c_len])
    for n_step, batch in enumerate(test_loader):
        ids, x, mask, y = batch
        x, mask = x.to(device), mask.to(device)
        outputs = model(x, attention_mask=mask)
        if args["model"] == "bert":
            outputs = outputs[0]
        _, labels = torch.max(outputs, dim=1)
        batch_size = labels.size(0)
        for i in range(batch_size):
            confusion_matrix[y[i].item()][labels[i].item()] += 1
    for k, v in constants.categories.items():
        p = confusion_matrix[v, v] / confusion_matrix[:, v].sum()
        r = confusion_matrix[v, v] / confusion_matrix[v, :].sum()
        f1 = 2 * p * r / (p + r + 0.001)
        print("{}: F1:{:.4f}, P:{:.4f}, R:{:.4f}".format(k, f1, p, r))

else:
    os.makedirs(checkpoint_dir) if not os.path.exists(checkpoint_dir) else None
    with open(os.path.join(checkpoint_dir, "args.json"), mode="w") as f:
        json.dump(args, f)
    train_loader, val_loader = prepare_loader(data["train"], args, mode="single")
    optimization_steps = args["epoch"]
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args["learning_rate"])
    scheduler = WarmupLinearSchedule(optimizer, optimization_steps // 10, optimization_steps)
    loss_fct = nn.CrossEntropyLoss()
    model = model.to(device)
    i2c = {v: k for k, v in constants.categories.items()}
    best_f1 = 0
    for epoch in range(1, args["epoch"] + 1):
        model.train()
        total_loss = 0
        for n_step, batch in enumerate(train_loader):
            ids, x, mask, y = batch
            x, mask, y = x.to(device), mask.to(device), y.to(device)
            optimizer.zero_grad()
            model.zero_grad()
            outputs = model(x, attention_mask=mask)
            if args["model"] == "bert":
                outputs = outputs[0]
            loss = loss_fct(outputs, y.view(-1))
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        scheduler.step()
        print("-----Epoch {}, Train Loss: {:.4f}----".format(epoch, total_loss/len(train_loader)))
        model.eval()
        # gt, pred
        confusion_matrix = np.zeros([constants.c_len, constants.c_len])
        for n_step, batch in enumerate(val_loader):
            ids, x, mask, y = batch
            x, mask = x.to(device), mask.to(device)
            outputs = model(x, attention_mask=mask)
            if args["model"] == "bert":
                outputs = outputs[0]
            labels = outputs.cpu().detach()
            _, labels = torch.max(labels, dim=1)
            batch_size = labels.size(0)
            for i in range(batch_size):
                confusion_matrix[y[i].item()][labels[i].item()] += 1
        f_value = 0
        for k, v in constants.categories.items():
            p = confusion_matrix[v, v] / confusion_matrix[:, v].sum()
            r = confusion_matrix[v, v] / confusion_matrix[v, :].sum()
            f1 = 2 * p * r / (p + r + 0.001)
            f_value += f1
        f_value /= constants.c_len
        print("f1: {:.4f}".format(f_value))
        if best_f1 < f_value:
            best_f1 = f_value
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "model.bin"))
