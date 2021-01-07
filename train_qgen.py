# -*- coding: utf-8 -*-
import torch
import torch.nn
import os
from utils.util import create_logger
from process_data.tokenizer import GWTokenizer
import gzip
import json
from utils.evaluate_tools import bleu_score, f1_score, generate_f1_score, multi_f1_score
from utils.scheduler import MyMSSchedule
from arguments.qgen_args import qgen_arguments


def train(model, args, train_loader, val_loader, param_file):
    optimizer = torch.optim.Adam(model.parameters(), args["lr"])
    scheduler = MyMSSchedule(optimizer, 3, [10, 20])
    best = 100
    stop_flag = 0
    for epoch in range(1, args["epoch_num"]+1):
        logger.info("------ Epoch {0} ------".format(epoch))
        model.train()
        train_loss = 0.0
        for batch_num, batch in enumerate(train_loader):
            optimizer.zero_grad()
            model.zero_grad()
            cls_loss, gen_loss = model(batch)
            loss = cls_loss + gen_loss
            if "cls" in args["task"]:
                cls_loss = cls_loss.item()
            if "gen" in args["task"]:
                gen_loss = gen_loss.item()
            loss.backward()
            # Clip gradients: gradients are modified in place
            _ = torch.nn.utils.clip_grad_norm_(model.parameters(), args["clip_val"])
            optimizer.step()
            train_loss += loss.item()
            if (batch_num+1) % args["display_step"] == 0:
                logger.info("Step {}/{}, Loss: {:.4f}, cls: {:.4f}, gen: {:.4f}".format(
                    epoch, batch_num, train_loss/batch_num, cls_loss, gen_loss))
        scheduler.step()
        train_loss = train_loss / len(train_loader)
        logger.info("average loss: {:.4f} lr: {:.7f}".format(train_loss, optimizer.param_groups[0]['lr']))

        # evaluate
        # bleu, f1, *_ = evaluate(model, valloader, args["task"])
        # logger.info("Eval: epoch {}, Average F1: {:.4f}, BLEU: {:.4f}".format(epoch, f1, bleu))
        # cur = bleu if "gen" in args["task"] else f1

        eval_loss = evaluate_loss(model, val_loader)
        logger.info("Eval: epoch {}, Loss: {:.4f}".format(epoch, eval_loss))
        cur = eval_loss
        if best > cur:
            torch.save(model.state_dict(), param_file)
            logger.info("save model to {}".format(param_file))
            best = min(best, cur)
            stop_flag = 0
        else:
            stop_flag += 1
            if 0 < args["early_stop"] <= stop_flag:
                logger.info("==================early stopping===================")
                break


def evaluate(model, val_loader, task, multi_cate):
    tokenizer = val_loader.dataset.tokenizer
    model.eval()
    game_ids, game_turns = [], []
    hys, refs, ref_cates, result_cates = [], [], [], []
    bleu, f1 = 0, 0
    for batch_num, batch in enumerate(val_loader):
        *_, y_cate, y, g_ids, g_turns = batch
        result, labels = model.generate(batch)
        game_ids.extend(g_ids)
        game_turns.extend(g_turns)
        if "cls" in task and labels is not None:
            result_cates.extend(labels.cpu().detach().tolist())
        else:
            result_cates.extend(y_cate.to(torch.long).tolist())
        ref_cates.extend(y_cate.to(torch.long).tolist())
        y = y.tolist()
        if "gen" in task:
            for i in range(len(y)):
                hys.append(tokenizer.decode(result[i]))
                refs.append(tokenizer.decode(y[i]))
        else:
            for i in range(len(y)):
                hys.append(tokenizer.decode(y[i]))
                refs.append(tokenizer.decode(y[i]))
    if "gen" in task:
        assert len(hys) == len(refs)
        bleu = bleu_score(hys, refs)
    if "cls" in task:
        assert len(ref_cates) == len(result_cates)
        f1 = multi_f1_score(result_cates, ref_cates) if multi_cate else f1_score(result_cates, ref_cates)
    return bleu, f1, [game_ids, game_turns, hys, refs, ref_cates, result_cates]


def evaluate_loss(model, val_loader):
    model.eval()
    total_loss = 0
    for batch_num, batch in enumerate(val_loader):
        cls_loss, gen_loss = model(batch)
        loss = cls_loss + gen_loss
        total_loss += loss.item()
    avg_loss = total_loss / len(val_loader)
    return avg_loss


def test(model, args, test_loader, param_file):
    model.load_state_dict(torch.load(param_file))  # load模型
    output_file = os.path.join(os.path.dirname(param_file), "output.json")
    print("load model from {}".format(param_file))
    model.eval()

    logger.info("=================load model and test======================")
    logger.info("total test batch num: {}".format(len(test_loader)))
    # test_loss = evaluate_loss(model, test_loader)
    # logger.info("Test Loss: {:.4f}".format(test_loss))
    bleu, f1, results = evaluate(model, test_loader, task=args["task"], multi_cate=args["multi_cate"])
    # hys, y_cates, result_cates = results[2], results[4], results[5]
    # generate_f1_score(hys, y_cates, result_cates, args)
    logger.info("blue score: {:.3f}, F1 score: {:.3f}".format(bleu, f1))
    outputs = {}
    games = new_get_games()
    for game_id, game_turn, hy, ref, y_cate, result_cate in zip(*results):
        if game_id not in outputs:
            outputs[game_id] = {}
        outputs[game_id][game_turn] = {
            "pred": hy,
            "pred_cate": result_cate,
            "ref_cate": y_cate,
            "answer": games[game_id]["qas"][game_turn]["answer"],
            "ref": games[game_id]["qas"][game_turn]["question"],
            "image_name": games[game_id]["image"]["file_name"]
        }
    with open(output_file, mode="w") as f:
        json.dump(outputs, f, indent=2)


def new_get_games():
    games = {}
    with gzip.open("./../data/guesswhat.test.jsonl.gz") as f:
        for line in f:
            line = line.decode("utf-8")
            game = json.loads(line.strip('\n'))
            if game["id"] in games:
                print("wrong")
                break
            games[game["id"]] = game
    return games


def main(args):
    param_file = save_path.format("params.pth.tar")
    data_dir = "./../data/"
    model_name = args["model"].lower()
    tokenizer = GWTokenizer('./../data/dict.json')
    if model_name == "cat_base":
        from models.qgen.qgen_cat_base import QGenNetwork
        from data_provider.qgen_dataset import prepare_dataset
    elif model_name == "hrnn":
        from models.qgen.qgen_hrnn import QGenNetwork
        from data_provider.qgen_dataset import prepare_dataset
    elif model_name == "cat_accu":
        from models.qgen.qgen_cat_accu import QGenNetwork
        from data_provider.qgen_dataset import prepare_dataset
    elif model_name == "cat_attn":
        from models.qgen.qgen_cat_attn import QGenNetwork
        from data_provider.qgen_dataset import prepare_dataset
    else:
        print(model_name)
        from models.qgen.qgen_baseline import QGenNetwork
        from data_provider.qgen_baseline_dataset import prepare_dataset
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args["option"] == "train":
        if args["image"] is False and args["object"] is False:
            print("default object")
            args["object"] = True
            args["image_arch"] = "rcnn"
            args["image_dim"] = 2048
        with open(save_path.format("args.json"), mode="w") as f:
            json.dump(args, f, indent=2, ensure_ascii=False)
        logger.info(args)
        model = QGenNetwork(args, tokenizer, device).to(device)
        train_loader, val_loader = prepare_dataset(data_dir, "train", args, tokenizer)
        train(model, args, train_loader, val_loader, param_file)
    else:
        with open(save_path.format("args.json"), mode="r") as f:
            saved_args = json.load(f)
            saved_args["option"] = "test"
        args = saved_args
        logger.info(args)
        model = QGenNetwork(args, tokenizer, device).to(device)
        testloader = prepare_dataset(data_dir, "test", args, tokenizer)
        test(model, args, testloader, param_file)


if __name__ == "__main__":
    parser = qgen_arguments()
    flags, unknown = parser.parse_known_args()
    flags = vars(flags)
    model_dir = "./../out/qgen/" + flags["name"]
    os.makedirs(model_dir) if not os.path.exists(model_dir) else None
    save_path = model_dir + "/{}"
    logger = create_logger(save_path.format('train.log'), "w")
    main(flags)
