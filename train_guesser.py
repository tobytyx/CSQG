# -*- coding: utf-8 -*-
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional
from utils.util import create_logger
from utils.calculate_util import calculate_accuracy
from process_data.tokenizer import GWTokenizer
from arguments.guesser_args import guesser_arguments
from data_provider.guesser_dataset import prepare_dataset
from models.guesser.baseline_model import GuesserNetwork
from utils.scheduler import MyMSSchedule


def train(model, args, train_loader, val_loader, param_file):
    optimizer = torch.optim.Adam(model.parameters(), args["lr"])
    scheduler = MyMSSchedule(optimizer, 3, [])
    best_err = 1
    stop_flag = 0
    for epoch in range(1, args["epoch_num"]+1):
        logger.info("------ Epoch {0} ------".format(epoch))
        model.train()
        train_accuracy = 0.0
        train_loss = 0.0
        for batch_num, batch in enumerate(train_loader):
            optimizer.zero_grad()
            model.zero_grad()
            output = model(batch)
            target = batch[-1].to(model.device).long()  # target object index
            loss = torch.nn.functional.cross_entropy(output, target)
            loss.backward()
            # Clip gradients: gradients are modified in place
            _ = nn.utils.clip_grad_norm_(model.parameters(), args["clip_val"])
            optimizer.step()
            _, accuracy = calculate_accuracy(output, target)
            train_loss += loss.item()
            train_accuracy += accuracy
            if (batch_num+1) % args["display_step"] == 0:
                logger.info("{0}/{1} loss: {2:.3} error: {3:.3}".format(
                    epoch, batch_num, loss, 1-accuracy / batch_num)
                )
        scheduler.step()
        train_loss = train_loss / len(train_loader)
        train_error = 1 - train_accuracy / len(train_loader)
        logger.info("average loss: {:.4f}, error: {:.3} lr: {:.7f}".format(
            train_loss, train_error, optimizer.param_groups[0]['lr']))
        eval_error = evaluate(model, val_loader)
        logger.info("Eval: epoch {} , Error: {:.4f}".format(epoch, eval_error))
        if eval_error < best_err:
            # 当前eval集合上的结果的错误率低于当前最后的error，则保存模型
            torch.save(model.state_dict(), param_file)
            logger.info("save model to {}".format(param_file))
            best_err = eval_error
            stop_flag = 0
        else:
            stop_flag += 1
            if 0 < args["early_stop"] <= stop_flag:
                logger.info("==================early stopping===================")
                break


def evaluate(model, val_loader):
    eval_accuracy = 0.0
    model.eval()
    for batch_num, batch in enumerate(val_loader):
        output = model(batch)
        target = batch[-1].to(model.device).long()  # answer
        _, accuracy = calculate_accuracy(output, target)
        eval_accuracy += accuracy
    error_rate = 1 - eval_accuracy / len(val_loader)
    return error_rate


def test(model, test_loader, param_file):
    print("load model from {}".format(param_file))
    model.load_state_dict(torch.load(param_file))  # load模型
    logger.info("=================load model and test======================")
    test_error = evaluate(model, test_loader)
    logger.info("test: average error: {}".format(test_error))


def main(args):
    param_file = save_path.format("params.pth.tar")
    data_dir = "./../data/"
    tokenizer = GWTokenizer('./../data/dict.json')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GuesserNetwork(args, tokenizer, device).to(device)
    if args["option"] == "train":
        with open(save_path.format("args.json"), mode="w") as f:
            json.dump(args, f, indent=2, ensure_ascii=False)
        logger.info(args)
        train_loader, val_loader = prepare_dataset(data_dir, "train", args, tokenizer)
        train(model, args, train_loader, val_loader, param_file)
    else:
        with open(save_path.format("args.json"), mode="r") as f:
            saved_args = json.load(f)
            saved_args["option"] = "test"
        args = saved_args
        logger.info(args)
        test_loader = prepare_dataset(data_dir, "test", args, tokenizer)
        test(model, test_loader, param_file)


if __name__ == "__main__":
    parser = guesser_arguments()
    flags, unknown = parser.parse_known_args()
    flags = vars(flags)
    model_dir = "./../out/guesser/" + flags["name"]
    os.makedirs(model_dir) if not os.path.exists(model_dir) else None
    save_path = model_dir + "/{}"
    logger = create_logger(save_path.format('train.log'), "w")
    main(flags)
