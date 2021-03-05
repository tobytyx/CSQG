# -*- coding: utf-8 -*-
import torch
import os
import json
import logging
from models.loop.looper import Looper
from models.loop.looper_rl import Looper as RL_Looper
from arguments.looper_args import looper_arguments, rl_looper_arguments
from process_data.tokenizer import GWTokenizer
from models.guesser.guesser_wrapper import GuesserWrapper
from models.oracle.oracle_wrapper import OracleWrapper
from models.qgen.qgen_wrapper import QuestionWrapper


def main():
    parser = looper_arguments()
    args, _ = parser.parse_known_args()
    args = vars(args)
    print(args)
    data_dir = "./../data"
    tokenizer = GWTokenizer("./../data/dict.json")
    out_dir = "./../out/"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    oracle = OracleWrapper(data_dir, out_dir, args["oracle_name"], tokenizer, device)
    guess = GuesserWrapper(data_dir, out_dir, args["guesser_name"], tokenizer, device)
    question = QuestionWrapper(data_dir, out_dir, args["qgen_name"], tokenizer, device)
    loop = Looper(data_dir=data_dir, oracle=oracle, guesser=guess, question=question, args=args)
    _, success_rate = loop.eval(
        option=args["option"], out_dir=out_dir, store=not args["no_store"], visualize=args["visualize"])
    result_file = os.path.join(out_dir, "games", "test.json")
    turns = "{}turns".format(args["max_turn"])
    models_name = ",".join([args["option"], turns, args["qgen_name"], args["oracle_name"], args["guesser_name"]])
    print("model_name: ", models_name)
    res = {}
    if os.path.exists(result_file):
        with open(result_file, mode="r") as f:
            res = json.load(f)
    res[models_name] = round(success_rate, 4)
    with open(result_file, mode="w") as f:
        json.dump(res, f, indent=2)


def greedy_main():
    parser = rl_looper_arguments()
    args, _ = parser.parse_known_args()
    args = vars(args)
    print(args)
    logger = logging.getLogger()
    args["cate_rl"] = "cls" in args["rl_task"]
    args["gen_rl"] = "gen" in args["rl_task"]
    logger.info(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = "./../data"
    looper = RL_Looper(data_dir, args, device, logger)
    success_rate, failed, success = looper.eval()
    result_file = "../out/games/test_greedy.json"
    res = {}
    if os.path.exists(result_file):
        with open(result_file, mode="r") as f:
            res = json.load(f)
    models_name = ",".join(
        [args["option"], str(args["max_turn"])+"turns", args["qgen_name"], args["oracle_name"], args["guesser_name"]])
    res[models_name] = round(success_rate, 4)
    print("successful rate: {:.2f}%".format(success_rate*100))
    with open(result_file, mode="w") as f:
        json.dump(res, f, indent=2)


if __name__ == '__main__':
    greedy_main()
