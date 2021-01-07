# -*- coding: utf-8 -*-
import argparse


def looper_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--option", type=str, default="new_pictures", choices=["new_pictures", "new_objects"])
    parser.add_argument("--batch_size", type=int, default=32, help="batch size for batched generating new games")
    parser.add_argument("--max_turn", type=int, default=5, help="max turn of each game")
    parser.add_argument("--qgen_name", type=str, default="baseline")
    parser.add_argument("--oracle_name", type=str, default="baseline")
    parser.add_argument("--guesser_name", type=str, default="baseline")
    parser.add_argument("--no_store", default=False, action="store_true")
    parser.add_argument("--visualize", default=False, action="store_true")
    return parser


def rl_looper_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="default")
    parser.add_argument("--log_step", type=int, default=200)
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128, help="batch size for batched generating new games")
    parser.add_argument("--max_turn", type=int, default=5, help="max turn of each game")
    parser.add_argument("--rl_task", type=str, default="gen", choices=["cls", "gen", "cls_gen"])
    parser.add_argument("--qgen_name", type=str, default="baseline")
    parser.add_argument("--oracle_name", type=str, default="baseline")
    parser.add_argument("--guesser_name", type=str, default="baseline")
    parser.add_argument("--no_store", default=False, action="store_true")
    parser.add_argument("--visualize", default=False, action="store_true")
    parser.add_argument("--sample_rate", default=1., type=float, help="sample rate for over fit")
    parser.add_argument("--lr", type=str, default=1e-4)
    return parser
