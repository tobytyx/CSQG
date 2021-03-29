# -*- coding: utf-8 -*-
import torch
import os
import json
from utils import constants
from models.guesser.baseline_model import GuesserNetwork
from process_data.tokenizer import GWTokenizer
from data_provider.gw_dataset import ImageProvider
from data_provider.loop_rl_dataset import get_bbox


qgen_name = "cat_v_rcnn_cls_prior_cate"
guesser_name = "baseline"
data_dir = "./../data"

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
tokenizer = GWTokenizer("./../data/dict.json")
with open(os.path.join("../out/qgen", qgen_name, "args.json")) as f:
    qgen_args = json.load(f)
with open(os.path.join("../out/guesser", guesser_name, "args.json")) as f:
    guesser_args = json.load(f)
guesser_model = GuesserNetwork(guesser_args, tokenizer, device).to(device)
guesser_model.load_state_dict(
    torch.load(os.path.join("../out/guesser", guesser_name, "params.pth.tar"), map_location=device))
guesser_model.eval()

answer_dict = {0: 'Yes', 1: 'No', 2: 'N/A'}
token_to_answer_idx = {
    tokenizer.yes_token: 0,
    tokenizer.no_token: 1,
    tokenizer.non_applicable_token: 2
}
if qgen_args["model"] == "cat_base":
    from models.qgen.qgen_cat_base import QGenNetwork
elif qgen_args["model"] == "cat_accu":
    from models.qgen.qgen_cat_accu import QGenNetwork
elif qgen_args["model"] == "hrnn":
    from models.qgen.qgen_hrnn import QGenNetwork
else:
    from models.qgen.qgen_baseline import QGenNetwork
qgen_model = QGenNetwork(args=qgen_args, tokenizer=tokenizer, device=device).to(device)
qgen_model.load_state_dict(
    torch.load(os.path.join("../out/qgen", qgen_name, "params.pth.tar"), map_location=device), strict=False)
qgen_model.eval()

object_builder = ImageProvider(
    os.path.join(data_dir, "features", "rcnn", "size,rcnn_arch,224.txt"), "file", "rcnn")
image_builder = ImageProvider(
    os.path.join(data_dir, "features", "vgg16", "fc8", "image.pkl"), "feature", "vgg16")


def prepare_qgen(question, answers, category=None, pre_questions=None,
                 pre_qs_lens=None, pre_q_indexes=None, pre_answers=None, pre_categories=None):
    """

    :param question: B * len
    :param answers: B
    :param category: B / B * cate_len / None
    :param pre_questions: B * total_len
    :param pre_qs_lens: B
    :param pre_q_indexes: B * turn
    :param pre_answers: B * turn
    :param pre_categories: B * turn / B * turn * cate_len / None
    :return:
    """
    bsz = question.size(0)
    multi_cate = qgen_args["multi_cate"]
    cur_len = torch.sum(torch.ne(question, 0) * torch.ne(question, constants.EOS), dim=1).to(
        dtype=torch.long, device=device)
    if pre_questions is None:
        max_len = torch.max(cur_len).item()
        cur_questions = torch.zeros(bsz, max_len+1, dtype=torch.long, device=device)
        for i in range(bsz):
            qs_len = cur_len[i].item()
            cur_questions[i, :qs_len] = question[i, :qs_len]
            cur_questions[i, qs_len] = constants.EOS
        cur_qs_lens = cur_len + 1
        cur_q_indexes = cur_len.unsqueeze(1)
    else:
        max_len = torch.max(pre_qs_lens + cur_len).item()
        cur_questions = torch.zeros(bsz, max_len, dtype=torch.long, device=device)
        for i in range(bsz):
            pre_qs_len = pre_qs_lens[i].item()
            qs_len = cur_len[i].item()
            cur_questions[i, :pre_qs_len-1] = pre_questions[i, :pre_qs_len-1]
            cur_questions[i, pre_qs_len-1:pre_qs_len+qs_len-1] = question[i, :qs_len]
            cur_questions[i, pre_qs_len+qs_len-1] = constants.EOS
        cur_qs_lens = pre_qs_lens + cur_len
        cur_q_index = pre_q_indexes[:, -1] + cur_len
        cur_q_indexes = torch.cat([pre_q_indexes, cur_q_index.unsqueeze(1)], dim=1)
    cur_answers = answers.unsqueeze(1)
    if pre_answers is not None:
        cur_answers = torch.cat([pre_answers, cur_answers], dim=1)

    if category is None:
        if multi_cate:
            category = torch.tensor([[0, 0, 0, 0] for _ in range(bsz)], dtype=torch.float, device=device)
        else:
            category = torch.tensor([3] * bsz, dtype=torch.long, device=device)
    cur_categories = category.unsqueeze(1)
    if pre_categories is not None:
        cur_categories = torch.cat([pre_categories, cur_categories], dim=1)
    return cur_questions, cur_qs_lens, cur_q_indexes, cur_answers, cur_categories


def prepare_dialogues(question, answer, pre_dials=None, pre_dial_lens=None):
    """
    :param question: B * len
    :param answer: B
    :param pre_dials: B * total_len
    :param pre_dial_lens: B
    :return:
    """
    bsz = question.size(0)
    cur_q_len = torch.sum(torch.ne(question, 0) * torch.ne(question, constants.EOS), dim=1).to(
        dtype=torch.long, device=device)
    answer_token = [answer_dict[a] for a in answer.cpu().tolist()]
    answer = []
    for token in answer_token:
        answer.extend(tokenizer.apply(token, is_answer=True))

    if pre_dials is None:
        max_dial_len = torch.max(cur_q_len).item() + 1
        dials = torch.zeros(bsz, max_dial_len, dtype=torch.long, device=device)
        dial_lens = cur_q_len + 1
        for i in range(bsz):
            q_len = cur_q_len[i].item()
            dials[i, :q_len] = question[i, :q_len]
            dials[i, q_len] = answer[i]
    else:
        max_dial_len = torch.max(cur_q_len+pre_dial_lens).item() + 1
        dials = torch.zeros(bsz, max_dial_len, dtype=torch.long, device=device)
        dial_lens = pre_dial_lens + cur_q_len + 1
        for i in range(bsz):
            pre_dial_len = pre_dial_lens[i].item()
            q_len = cur_q_len[i].item()
            dials[i, :pre_dial_len] = pre_dials[i, :pre_dial_len]
            dials[i, pre_dial_len:pre_dial_len+q_len] = question[i, :q_len]
            dials[i, pre_dial_len+q_len] = answer[i]
    return dials, dial_lens


def qgen(session):
    img = object_builder.load_feature(session["img_name"])
    qgen_img = img["att"]
    qgen_bbox = get_bbox(img["pos"])
    q_imgs = torch.tensor([qgen_img], dtype=torch.float)
    q_bbox = torch.tensor([qgen_bbox], dtype=torch.float)
    if session["cache"]["turn"] == 0:
        question = torch.ones(1, 1, dtype=torch.long, device=device) * constants.EOS
        answer = torch.tensor([2], dtype=torch.long, device=device)
        questions, qs_lens, q_indexes, answers, categories = prepare_qgen(question, answer)
    else:
        questions = session["cache"]["questions"]
        qs_lens = session["cache"]["qs_lens"]
        q_indexes = session["cache"]["q_indexes"]
        answers = session["cache"]["answers"]
        categories = session["cache"]["categories"]
    
    turns = torch.tensor([q_indexes.size(1)], dtype=torch.long, device=device)
    with torch.no_grad():
        _, _, category, question = qgen_model.pg_forward(
            questions, qs_lens, q_indexes, answers, categories, turns, q_imgs, q_bbox
        )
    session["cache"]["category"] = category
    session["cache"]["question"] = question

    question = question[0].detach().cpu().tolist()
    question = tokenizer.decode(question)
    return question
    

def oracle(session, answer):
    answer = torch.tensor([answer], dtype=torch.long, device=device)
    if session["cache"]["turn"] == 0:
        dials, dial_lens = prepare_dialogues(session["cache"]["question"], answer)
        questions, qs_lens, q_indexes, answers, categories = prepare_qgen(
            session["cache"]["question"], answer, session["cache"]["category"])
    else:
        dials, dial_lens = prepare_dialogues(
            session["cache"]["question"], answer, session["cache"]["dials"], session["cache"]["dial_lens"])
        questions, qs_lens, q_indexes, answers, categories = prepare_qgen(
            session["cache"]["question"], answer, session["cache"]["category"],
            session["cache"]["questions"], session["cache"]["qs_lens"],
            session["cache"]["q_indexes"], session["cache"]["answers"], session["cache"]["categories"])
    session["cache"]["dials"] = dials
    session["cache"]["dial_lens"] = dial_lens
    session["cache"]["questions"] = questions 
    session["cache"]["qs_lens"] = qs_lens
    session["cache"]["q_indexes"] = q_indexes
    session["cache"]["answers"] = answers
    session["cache"]["categories"] = categories

    session["cache"]["turn"] += 1


def guess(session, img_id, cats, spas):
    g_imgs = image_builder.load_feature(img_id)
    g_imgs = torch.tensor([g_imgs], dtype=torch.float)
    g_obj_mask = torch.tensor([[1] * len(cats)], dtype=torch.long)
    g_cats = torch.tensor([cats], dtype=torch.long)
    g_spas = torch.tensor([spas], dtype=torch.float)
    batch = [session["cache"]["dials"], session["cache"]["dial_lens"],
             g_imgs, g_obj_mask, g_cats, g_spas]
    with torch.no_grad():
        predict = guesser_model(batch)
    predict = torch.argmax(predict, dim=1)
    predict = predict[0].item()
    return predict
