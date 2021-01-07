# -*- coding: utf-8 -*-
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from utils.constants import c_len, categories, BERT_BASE_UNCASED_PATH
import torch
import os
import json
import numpy as np
from transformers import BertTokenizer
from data_labeling.classifier import BertClassify
bleu_smooth = SmoothingFunction()


def bleu_score(hypothesis, references):
    refs = []
    hyps = []
    assert len(hypothesis) == len(references)
    for i in range(len(hypothesis)):
        ref = references[i].split(' ')
        hyp = hypothesis[i].split(' ')
        refs.append([ref])
        hyps.append(hyp)
    score = corpus_bleu(refs, hyps, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=bleu_smooth.method2)
    return score


def punch_label(model, result, device, th=0.4):
    max_b = 8
    total_b = len(result)
    cur = 0
    total_cates = []
    while cur < total_b:
        xs = result[cur: cur+max_b]
        max_len = max(len(x) for x in xs)
        seqs = torch.tensor([x + [0] * (max_len - len(x)) for x in xs], dtype=torch.long, device=device)
        mask = torch.tensor([[1] * len(x) + [0] * (max_len - len(x)) for x in xs], dtype=torch.long, device=device)
        with torch.no_grad():
            outputs = model(seqs, attention_mask=mask)
            labels = torch.sigmoid(outputs[0]).cpu().detach()
        cates = (labels > th).tolist()
        total_cates.extend(cates)
        cur += max_b
    return total_cates


def multi_f1_score(pred_cates, ref_cates):
    """
    计算f1值
    :param pred_cates:
    :param ref_cates:
    :return f1: F1 score
    """
    tp, tn, fp, fn = 0, 0, 0, 0
    # accumulate ngram statistics
    for hy_cate, ref_cate in zip(pred_cates, ref_cates):
        for j in range(len(hy_cate)):
            if hy_cate[j] == 1 and ref_cate[j] == 1:
                tp += 1
            elif hy_cate[j] == 1 and ref_cate[j] == 0:
                fp += 1
            elif hy_cate[j] == 0 and ref_cate[j] == 0:
                tn += 1
            else:
                fn += 1
    precision = tp / (tp + fp + 0.001)
    recall = tp / (tp + fn + 0.001)
    f1 = 2 * precision * recall / (precision + recall + 0.001)
    return f1


def f1_score(pred_cates, ref_cates):
    confusion_matrix = np.zeros([c_len, c_len])
    for pred, gt in zip(pred_cates, ref_cates):
        confusion_matrix[gt, pred] += 1
    f_value = 0
    total_sum = confusion_matrix.sum()
    print("***Eval per category***")
    for k, v in categories.items():
        p = confusion_matrix[v, v] / (confusion_matrix[:, v].sum() + 0.001)
        r = confusion_matrix[v, v] / (confusion_matrix[v, :].sum() + 0.001)
        f1 = 2 * p * r / (p + r + 0.001)
        f_value += f1 * (confusion_matrix[v, :].sum() / total_sum)
        print("{}: F1:{:.4f}, P:{:.4f}, R:{:.4f}".format(k, f1, p, r))
    return f_value


def generate_f1_score(hys, ref_cates, result_cates, args):
    checkpoint_dir = "../out/bert_cnn"
    with open(os.path.join(checkpoint_dir, "args.json")) as f:
        bert_args = json.load(f)
    device = torch.device("cuda")
    model = BertClassify(bert_args)
    model.load_state_dict(torch.load(os.path.join(checkpoint_dir, "model.bin")))
    model = model.to(device)
    model.eval()
    tokenizer = BertTokenizer.from_pretrained(BERT_BASE_UNCASED_PATH)
    hy_cates = []
    batch_size = 8
    step = 0
    decoder_true, decode_total = 0, 0
    while step < len(hys):
        xs = []
        for hy in hys[step: step+batch_size]:
            x = ["[CLS]"] + tokenizer.tokenize(hy) + ["[SEP]"]
            x = tokenizer.convert_tokens_to_ids(x)
            xs.append(x)
        max_len = max(len(x) for x in xs)
        x_batch = torch.tensor([x + [0] * (max_len - len(x)) for x in xs], dtype=torch.long, device=device)
        x_mask = torch.tensor([[1] * len(x) + [0] * (max_len - len(x)) for x in xs], dtype=torch.long, device=device)
        outputs = model(x_batch, attention_mask=x_mask)
        if args["multi_cate"]:
            labels = torch.sigmoid(outputs)
            labels = (labels > args["th"]).cpu().detach().tolist()
        else:
            _, labels = torch.max(outputs, dim=1)
            labels = labels.cpu().detach().tolist()
        for i in range(len(labels)):
            decode_total += 1
            if labels[i] == result_cates[step+i]:
                decoder_true += 1
        hy_cates.extend(labels)
        step = step + batch_size
    print("decode acc: {:.4f}".format(decoder_true / (decode_total + 0.1)))
    print("sentence category: ")
    f_value = multi_f1_score(hy_cates, ref_cates) if args["multi_cate"] else f1_score(hy_cates, ref_cates)
    print("total sentence category F1: {:.4f}".format(f_value))
