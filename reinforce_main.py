# -*- coding: utf-8 -*-
import torch
import os
import json
from models.loop.looper_rl import Looper
from arguments.looper_args import rl_looper_arguments
from utils.util import create_logger


def main():
    parser = rl_looper_arguments()
    args, unknown = parser.parse_known_args()
    args = vars(args)
    save_path = os.path.join("../out/qgen", args["qgen_name"], args["name"])
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    log_path = os.path.join(save_path, "rl_train.log")
    logger = create_logger(log_path, "w")
    args["cate_rl"] = "cls" in args["rl_task"]
    args["gen_rl"] = "gen" in args["rl_task"]
    logger.info(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    data_dir = "./../data"
    looper = Looper(data_dir, args, device, logger)
    optimizer = torch.optim.Adam(looper.qgen_model.parameters(), lr=args["lr"])

    success_rate, failed, success = looper.eval()
    with open(os.path.join(save_path, "rl_fail_start.json"), mode="w") as f:
        json.dump(failed, f, indent=2)
    with open(os.path.join(save_path, "rl_success_start.json"), mode="w") as f:
        json.dump(success, f, indent=2)
    best_success_rate = success_rate
    logger.info("Init success_rate: {:.3f}%".format(success_rate * 100))
    failed_file = os.path.join(save_path, "rl_fail.json")
    success_file = os.path.join(save_path, "rl_success.json")
    for epoch in range(1, args["epoch"]):
        # reward_loss = looper.rl_train_epoch(optimizer)
        reward_loss = looper.rl_sample_reward_epoch(optimizer)
        print("epoch: {}, reward_loss: {:.4f}".format(epoch, reward_loss))
        if epoch % 2 == 0:
            success_rate, failed, success = looper.eval()
            with open(failed_file, mode="w") as f:
                json.dump(failed, f)
            with open(success_file, mode="w") as f:
                json.dump(success, f)
            logger.info("epoch: {}, success_rate: {:.3f}%".format(epoch, success_rate*100))
            if success_rate > best_success_rate:
                best_success_rate = success_rate
                torch.save(
                    looper.qgen_model.state_dict(),
                    os.path.join("../out/qgen", args["qgen_name"], "rl_model.bin"))
    success_rate, failed, success = looper.eval()
    with open(failed_file, mode="w") as f:
        json.dump(failed, f)
    with open(success_file, mode="w") as f:
        json.dump(success, f)
    result_file = os.path.join("../out/games", "test.json")
    turns = "{}turns".format(args["max_turn"])
    models_name = ",".join(["reinforce", turns, args["qgen_name"], args["oracle_name"], args["guesser_name"]])
    print("model_name: ", models_name)
    res = {}
    if os.path.exists(result_file):
        with open(result_file, mode="r") as f:
            res = json.load(f)
    res[models_name] = round(success_rate, 4)
    with open(result_file, mode="w") as f:
        json.dump(res, f, indent=2)


if __name__ == '__main__':
    main()
