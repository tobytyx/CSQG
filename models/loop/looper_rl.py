# -*- coding:utf-8 -*-
import json
import os
import torch
from torch.utils.data import DataLoader
from models.guesser.baseline_model import GuesserNetwork
from models.oracle.baseline_model import OracleNetwork
from process_data.tokenizer import GWTokenizer
from data_provider.loop_rl_dataset import LoopRlDataset, loop_rl_collate
from utils import constants
import random


class Looper(object):
    def __init__(self, data_dir, args, device, logger):
        oracle_name, guesser_name, qgen_name = args["oracle_name"], args["guesser_name"], args["qgen_name"]
        self.data_dir = data_dir
        self.device = device
        self.args = args
        self.tokenizer = GWTokenizer("./../data/dict.json")
        self.logger = logger
        with open(os.path.join("../out/oracle", oracle_name, "args.json")) as f:
            self.oracle_args = json.load(f)
        with open(os.path.join("../out/qgen", qgen_name, "args.json")) as f:
            self.qgen_args = json.load(f)
        with open(os.path.join("../out/guesser", guesser_name, "args.json")) as f:
            self.guesser_args = json.load(f)
        self.guesser_model = GuesserNetwork(self.guesser_args, self.tokenizer, self.device).to(self.device)
        self.guesser_model.load_state_dict(
            torch.load(os.path.join("../out/guesser", guesser_name, "params.pth.tar")))
        self.guesser_model.eval()

        self.oracle_model = OracleNetwork(self.oracle_args, self.tokenizer, self.device).to(self.device)
        self.oracle_model.load_state_dict(
            torch.load(os.path.join("../out/oracle", oracle_name, "params.pth.tar")))
        self.oracle_model.eval()
        self.answer_dict = {0: 'Yes', 1: 'No', 2: 'N/A'}
        self.token_to_answer_idx = {
            self.tokenizer.yes_token: 0,
            self.tokenizer.no_token: 1,
            self.tokenizer.non_applicable_token: 2
        }
        if self.qgen_args["model"] == "cat_base":
            from models.qgen.qgen_cat_base import QGenNetwork
        elif self.qgen_args["model"] == "cat_accu":
            from models.qgen.qgen_cat_accu import QGenNetwork
        elif self.qgen_args["model"] == "hrnn":
            from models.qgen.qgen_hrnn import QGenNetwork
        else:
            from models.qgen.qgen_baseline import QGenNetwork
        self.qgen_model = QGenNetwork(args=self.qgen_args, tokenizer=self.tokenizer, device=self.device).to(self.device)
        self.qgen_model.load_state_dict(
            torch.load(os.path.join("../out/qgen", qgen_name, "params.pth.tar")), strict=False)
        train_dataset = LoopRlDataset(
            data_dir, "train", self.qgen_args, self.oracle_args, self.guesser_args, self.tokenizer)
        # train_dataset.games = train_dataset.games[:100]
        self.train_loader = DataLoader(
            train_dataset,
            num_workers=4, collate_fn=loop_rl_collate, shuffle=True, batch_size=args["batch_size"]
        )
        self.gt_qas = {}
        for game in train_dataset.games:
            self.gt_qas[game.id] = {"questions": game.questions, "answers": game.answers}

        self.test_loader = DataLoader(
            LoopRlDataset(data_dir, "test", self.qgen_args, self.oracle_args, self.guesser_args, self.tokenizer),
            num_workers=4, collate_fn=loop_rl_collate, shuffle=False, batch_size=args["batch_size"]
        )

    def rl_train_epoch(self, optimizer):
        self.qgen_model.train()
        total_reward_loss = 0
        total_reward = 0
        last_reward = 0
        last_reward_loss = 0
        last_step = 0
        # total_baseline_loss = 0
        steps = 1
        for batch in self.train_loader:
            _, q_imgs, q_bbox, o_imgs, o_crops, o_cats, o_spas, g_imgs, g_obj_mask, g_cats, g_spas, targets = batch
            self.qgen_model.zero_grad()
            optimizer.zero_grad()
            reward, reward_loss = self.rl_step(
                q_imgs, q_bbox, o_imgs, o_crops, o_cats, o_spas, g_imgs, g_obj_mask, g_cats, g_spas, targets)
            reward_loss.backward()
            optimizer.step()

            total_reward_loss += reward_loss.item()
            total_reward += reward.mean().item()
            # total_baseline_loss += baseline_loss.item()
            if steps % self.args["log_step"] == 0:
                log_loss = (total_reward_loss - last_reward_loss) / (steps - last_step)
                log_reward = (total_reward - last_reward) / (steps - last_step)
                self.logger.info("Step {}, Loss {:.4f}, Reward: {}".format(
                    steps, log_loss, log_reward))
                last_step = steps
                last_reward = total_reward
                last_reward_loss = total_reward_loss
            steps += 1
        # total_baseline_loss /= (len(self.train_loader) * self.args["max_turn"])
        total_reward_loss /= len(self.train_loader)
        return total_reward_loss

    def rl_step(self, q_imgs, q_bbox, o_imgs, o_crops, o_cats, o_spas, g_imgs, g_obj_mask, g_cats, g_spas, targets):
        targets = targets.to(self.device)
        bsz = q_imgs.size(0)
        question = torch.ones(bsz, 1, dtype=torch.long, device=self.device) * constants.EOS
        answer = torch.tensor([2] * bsz, dtype=torch.long, device=self.device)
        dials, dial_lens = None, None
        questions, qs_lens, q_indexes, answers, categories = self.prepare_qgen(question, answer)
        total_gen_loss = []
        for turn in range(self.args["max_turn"]):
            turns = torch.tensor([q_indexes.size(1)] * bsz, dtype=torch.long, device=self.device)
            # qgen
            baseline, gen_loss, category, question = self.qgen_model.pg_forward(
                questions, qs_lens, q_indexes, answers, categories, turns, q_imgs, q_bbox
            )
            gen_loss = torch.sum(gen_loss, dim=1) * (0.9 ** turn)
            total_gen_loss.append(gen_loss)
            category, question = category.detach(), question.detach()
            # oracle
            q_lens = torch.sum(torch.ne(question, 0) * torch.ne(question, constants.EOS), dim=1).to(
                dtype=torch.long, device=self.device)
            batch = [question, q_lens, o_imgs, o_crops, o_cats, o_spas]
            answer = self.oracle_model(batch)
            answer = torch.argmax(answer, dim=1).detach()
            # guesser
            if turn == 0:
                dials, dial_lens = self.prepare_dialogues(question, answer)
            else:
                dials, dial_lens = self.prepare_dialogues(question, answer, dials, dial_lens)

            if turn == 0:
                questions, qs_lens, q_indexes, answers, categories = self.prepare_qgen(
                    question, answer, category)
            else:
                questions, qs_lens, q_indexes, answers, categories = self.prepare_qgen(
                    question, answer, category, questions, qs_lens, q_indexes, answers, categories)
        batch = [dials, dial_lens, g_imgs, g_obj_mask, g_cats, g_spas]
        predict = self.guesser_model(batch)
        predict = torch.argmax(predict, dim=1)
        # loss 没有log，每一轮的loss直接相加有点问题。完全平均不太对。越远的地方reward应该越强。
        reward = (predict == targets).to(dtype=torch.float, device=self.device).detach()
        # reward_score = torch.norm(reward - baseline) * gen_loss
        # reward_loss = torch.mean(torch.sum(reward_score, dim=1), dim=0)
        total_gen_loss = torch.stack(total_gen_loss)
        reward_loss = torch.mean(reward * torch.sum(total_gen_loss, dim=1), dim=0)
        # baseline_loss = torch.sum(torch.norm(reward - baseline))
        # loss = reward_loss + baseline_loss
        return reward, reward_loss

    def eval(self):
        self.qgen_model.eval()
        failed = {}
        success_dials = {}
        success_num, total_num = 0, 0
        with torch.no_grad():
            for batch in self.test_loader:
                game_ids, q_imgs, q_bbox, o_imgs, o_crops, o_cats, o_spas, g_imgs, g_obj_mask, g_cats, g_spas, targets = batch
                bsz = q_imgs.size(0)
                targets = targets.to(self.device)
                question = torch.ones(bsz, 1, dtype=torch.long, device=self.device) * constants.EOS
                answer = torch.tensor([2] * bsz, dtype=torch.long, device=self.device)
                dials, dial_lens = self.prepare_dialogues(question, answer)
                questions, qs_lens, q_indexes, answers, categories = self.prepare_qgen(question, answer)
                for turn in range(self.args["max_turn"]):
                    turns = torch.tensor([q_indexes.size(1)] * bsz, dtype=torch.long, device=self.device)
                    # print(questions.size(), qs_lens.size())
                    try:
                        _, _, category, question = self.qgen_model.pg_forward(
                            questions, qs_lens, q_indexes, answers, categories, turns, q_imgs, q_bbox
                        )
                    except Exception as e:
                        print(e)
                        questions = questions.detach().cpu().tolist()
                        qs_lens = qs_lens.detach().cpu().tolist()
                        with open("debug.json", mode="w") as f:
                            json.dump({"questions": questions, "qs_lens": qs_lens}, f)
                        raise
                    q_lens = torch.sum(torch.ne(question, 0) * torch.ne(question, constants.EOS), dim=1).to(
                        dtype=torch.long, device=self.device)
                    batch = [question, q_lens, o_imgs, o_crops, o_cats, o_spas]
                    answer = self.oracle_model(batch)
                    answer = torch.argmax(answer, dim=1)
                    if turn == 0:
                        dials, dial_lens = self.prepare_dialogues(question, answer)
                        questions, qs_lens, q_indexes, answers, categories = self.prepare_qgen(
                            question, answer, category)
                    else:
                        dials, dial_lens = self.prepare_dialogues(question, answer, dials, dial_lens)
                        questions, qs_lens, q_indexes, answers, categories = self.prepare_qgen(
                            question, answer, category, questions, qs_lens, q_indexes, answers, categories)
                batch = [dials, dial_lens, g_imgs, g_obj_mask, g_cats, g_spas]
                predict = self.guesser_model(batch)
                predict = torch.argmax(predict, dim=1)
                success = (predict == targets).to(torch.long)
                dials = dials.detach().cpu().tolist()
                dial_lens = dial_lens.detach().cpu().tolist()
                for i in range(success.size(0)):
                    if success[i].item() == 0:
                        game_id = game_ids[i]
                        dial = self.tokenizer.decode(dials[i][:dial_lens[i]])
                        failed[game_id] = dial
                    else:
                        game_id = game_ids[i]
                        dial = self.tokenizer.decode(dials[i][:dial_lens[i]])
                        success_dials[game_id] = dial
                success_num += torch.sum(success).item()
                total_num += predict.size(0)

        success_rate = success_num / total_num
        return success_rate, failed, success_dials

    def prepare_qgen(self, question, answers, category=None, pre_questions=None,
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
        multi_cate = self.qgen_args["multi_cate"]
        cur_len = torch.sum(torch.ne(question, 0) * torch.ne(question, constants.EOS), dim=1).to(
            dtype=torch.long, device=self.device)
        if pre_questions is None:
            max_len = torch.max(cur_len).item()
            cur_questions = torch.zeros(bsz, max_len+1, dtype=torch.long, device=self.device)
            for i in range(bsz):
                qs_len = cur_len[i].item()
                cur_questions[i, :qs_len] = question[i, :qs_len]
                cur_questions[i, qs_len] = constants.EOS
            cur_qs_lens = cur_len + 1
            cur_q_indexes = cur_len.unsqueeze(1)
        else:
            max_len = torch.max(pre_qs_lens + cur_len).item()
            cur_questions = torch.zeros(bsz, max_len, dtype=torch.long, device=self.device)
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
                category = torch.tensor([[0, 0, 0, 0] for _ in range(bsz)], dtype=torch.float, device=self.device)
            else:
                category = torch.tensor([3] * bsz, dtype=torch.long, device=self.device)
        cur_categories = category.unsqueeze(1)
        if pre_categories is not None:
            cur_categories = torch.cat([pre_categories, cur_categories], dim=1)
        return cur_questions, cur_qs_lens, cur_q_indexes, cur_answers, cur_categories

    def prepare_dialogues(self, question, answer, pre_dials=None, pre_dial_lens=None):
        """

        :param question: B * len
        :param answer: B
        :param pre_dials: B * total_len
        :param pre_dial_lens: B
        :return:
        """
        bsz = question.size(0)
        cur_q_len = torch.sum(torch.ne(question, 0) * torch.ne(question, constants.EOS), dim=1).to(
            dtype=torch.long, device=self.device)
        answer_token = [self.answer_dict[a] for a in answer.cpu().tolist()]
        answer = []
        for token in answer_token:
            answer.extend(self.tokenizer.apply(token, is_answer=True))

        if pre_dials is None:
            max_dial_len = torch.max(cur_q_len).item() + 1
            dials = torch.zeros(bsz, max_dial_len, dtype=torch.long, device=self.device)
            dial_lens = cur_q_len + 1
            for i in range(bsz):
                q_len = cur_q_len[i].item()
                dials[i, :q_len] = question[i, :q_len]
                dials[i, q_len] = answer[i]
        else:
            max_dial_len = torch.max(cur_q_len+pre_dial_lens).item() + 1
            dials = torch.zeros(bsz, max_dial_len, dtype=torch.long, device=self.device)
            dial_lens = pre_dial_lens + cur_q_len + 1
            for i in range(bsz):
                pre_dial_len = pre_dial_lens[i].item()
                q_len = cur_q_len[i].item()
                dials[i, :pre_dial_len] = pre_dials[i, :pre_dial_len]
                dials[i, pre_dial_len:pre_dial_len+q_len] = question[i, :q_len]
                dials[i, pre_dial_len+q_len] = answer[i]
        return dials, dial_lens

    def rl_sampling(self, game_ids, q_imgs, q_bbox, o_imgs, o_crops,
                    o_cats, o_spas, g_imgs, g_obj_mask, g_cats, g_spas, targets):
        sampling_dials = {}
        with torch.no_grad():
            targets = targets.to(self.device)
            bsz = q_imgs.size(0)
            question = torch.ones(bsz, 1, dtype=torch.long, device=self.device) * constants.EOS
            answer = torch.tensor([2] * bsz, dtype=torch.long, device=self.device)
            dials, dial_lens = None, None
            questions, qs_lens, q_indexes, answers, categories = self.prepare_qgen(question, answer)
            for turn in range(self.args["max_turn"]):
                turns = torch.tensor([q_indexes.size(1)] * bsz, dtype=torch.long, device=self.device)
                # qgen
                _, _, category, question = self.qgen_model.pg_forward(
                    questions, qs_lens, q_indexes, answers, categories, turns, q_imgs, q_bbox
                )
                category, question = category.detach(), question.detach()
                # oracle
                q_lens = torch.sum(torch.ne(question, 0) * torch.ne(question, constants.EOS), dim=1).to(
                    dtype=torch.long, device=self.device)
                q_lens_zero = (q_lens <= 0).to(dtype=torch.long, device=self.device)
                q_lens = q_lens + q_lens_zero
                batch = [question, q_lens, o_imgs, o_crops, o_cats, o_spas]
                answer = self.oracle_model(batch)
                answer = torch.argmax(answer, dim=1).detach()
                # guesser
                if turn == 0:
                    dials, dial_lens = self.prepare_dialogues(question, answer)
                else:
                    dials, dial_lens = self.prepare_dialogues(question, answer, dials, dial_lens)

                if turn == 0:
                    questions, qs_lens, q_indexes, answers, categories = self.prepare_qgen(
                        question, answer, category)
                else:
                    questions, qs_lens, q_indexes, answers, categories = self.prepare_qgen(
                        question, answer, category, questions, qs_lens, q_indexes, answers, categories)
            batch = [dials, dial_lens, g_imgs, g_obj_mask, g_cats, g_spas]
            predict = self.guesser_model(batch)
            predict = torch.argmax(predict, dim=1)
            success = (predict == targets).to(torch.long)
            dials = dials.detach().cpu().tolist()
            dial_lens = dial_lens.detach().cpu().tolist()

            # B * turn
            categories = categories.detach().cpu().tolist()
            for i in range(success.size(0)):
                game_id = game_ids[i]
                dial = dials[i][:dial_lens[i]]
                category = categories[i]
                questions, answers = split_dial(dial, self.token_to_answer_idx)
                sampling_dials[game_id] = {
                    "questions": questions, "answers": answers,
                    "successes": success[i].item(), "categories": category}
        return sampling_dials

    def rl_reward_step(self, q_imgs, q_bbox, all_questions, all_answers, successes, all_categories):
        bsz = q_imgs.size(0)
        question = torch.ones(bsz, 1, dtype=torch.long, device=self.device) * constants.EOS
        answer = torch.tensor([2] * bsz, dtype=torch.long, device=self.device)
        questions, qs_lens, q_indexes, answers, categories = self.prepare_qgen(question, answer)
        total_gen_loss = []
        total_cls_loss = []
        sos, eos, pad = self.tokenizer.start_token, self.tokenizer.stop_token, self.tokenizer.padding_token
        for turn in range(self.args["max_turn"]):
            turns = torch.tensor([q_indexes.size(1)] * bsz, dtype=torch.long, device=self.device)
            target_cate, targets = None, None
            if self.args["cate_rl"]:
                target_cate = all_categories[turn]

            targets = all_questions[turn]
            max_len = max([len(each) for each in targets])
            targets = [[sos] + target + [eos] + [pad] * (max_len - len(target)) for target in targets]
            targets = torch.tensor(targets, dtype=torch.long, device=self.device)

            cls_loss, gen_loss = self.qgen_model.pg_forward_with_target(
                questions, qs_lens, q_indexes, answers, categories,
                turns, q_imgs, q_bbox, target_cate, targets)
            if isinstance(gen_loss, torch.Tensor):
                gen_loss = torch.sum(gen_loss, dim=1)
            total_gen_loss.append(gen_loss)
            total_cls_loss.append(cls_loss)
            question = targets[:, 1:].detach()
            answer = torch.tensor(all_answers[turn], dtype=torch.long, device=self.device)
            if self.qgen_args["multi_cate"]:
                category = torch.tensor(all_categories[turn], dtype=torch.float, device=self.device)
            else:
                category = torch.tensor(all_categories[turn], dtype=torch.long, device=self.device)
            if turn == 0:
                questions, qs_lens, q_indexes, answers, categories = self.prepare_qgen(
                    question, answer, category)
            else:
                questions, qs_lens, q_indexes, answers, categories = self.prepare_qgen(
                    question, answer, category, questions, qs_lens, q_indexes, answers, categories)
        reward_gen_loss = 0
        reward_cls_loss = 0
        if self.args["gen_rl"]:
            total_gen_loss = torch.stack(total_gen_loss)  # turn * B
        if self.args["cate_rl"]:
            total_cls_loss = torch.stack(total_cls_loss)  # turn * B
        # total_gen_loss = torch.sum(total_gen_loss, dim=0)
        for i in range(bsz):
            gen_loss = 0
            cls_loss = 0
            for j in range(self.args["max_turn"]):
                for k in range(self.args["max_turn"]-j):
                    if self.args["gen_rl"]:
                        gen_loss = gen_loss + total_gen_loss[j, i] * (0.9 ** k)
                    if self.args["cate_rl"]:
                        cls_loss = cls_loss + total_cls_loss[j, i] * (0.9 ** k)
            if successes[i] == 1:
                reward_gen_loss = reward_gen_loss + gen_loss * 0.9
                reward_cls_loss = reward_cls_loss + cls_loss * 0.9
            else:
                reward_gen_loss = reward_gen_loss + gen_loss * -0.01
                reward_cls_loss = reward_cls_loss + cls_loss * -0.01
        reward_gen_loss /= bsz
        reward_cls_loss /= bsz
        return reward_cls_loss, reward_gen_loss

    def rl_sample_reward_epoch(self, optimizer):
        sample_rate = self.args["sample_rate"]
        sample_rate = max(0, min(1, sample_rate))
        # 采样
        sample_dials = {}
        self.qgen_model.eval()
        for batch in self.train_loader:
            game_ids, q_imgs, q_bbox, o_imgs, o_crops, o_cats, o_spas, g_imgs, g_obj_mask, g_cats, g_spas, targets = batch
            sampling_dial = self.rl_sampling(
                game_ids, q_imgs, q_bbox, o_imgs, o_crops, o_cats, o_spas, g_imgs, g_obj_mask, g_cats, g_spas, targets)
            sample_dials.update(sampling_dial)
        # 更新
        self.qgen_model.train()
        total_reward_loss = 0
        total_gen_loss, last_gen_loss = 0, 0
        total_cls_loss, last_cls_loss = 0, 0
        last_step, steps = 0, 1
        for batch in self.train_loader:
            if random.random() > sample_rate:
                continue
            game_ids, q_imgs, q_bbox, *_ = batch
            questions, answers, categories, successes = [], [], [], []
            for game_id in game_ids:
                dial = sample_dials[game_id]
                questions.append(dial["questions"])
                answers.append(dial["answers"])
                categories.append(dial["categories"])
                successes.append(dial["successes"])
            all_questions = list(zip(*questions))
            all_answers = list(zip(*answers))
            all_categories = list(zip(*categories))
            self.qgen_model.zero_grad()
            optimizer.zero_grad()
            reward_cls_loss, reward_gen_loss = self.rl_reward_step(
                q_imgs, q_bbox, all_questions, all_answers, successes, all_categories)
            reward_loss = 0
            if self.args["cate_rl"]:
                reward_loss += reward_cls_loss
                total_cls_loss += reward_cls_loss.item()
            if self.args["gen_rl"]:
                reward_loss += reward_gen_loss
                total_gen_loss += reward_gen_loss.item()
            reward_loss.backward()
            optimizer.step()
            total_reward_loss += reward_loss.item()
            if steps % self.args["log_step"] == 0:
                gen_loss = (total_gen_loss - last_gen_loss) / (steps - last_step)
                cls_loss = (total_cls_loss - last_cls_loss) / (steps - last_step)
                self.logger.info(
                    "Step {}, Gen Loss {:.4f}, Cls Loss {:.4f}".format(steps, gen_loss, cls_loss))
                last_step = steps
                last_gen_loss = total_gen_loss
                last_cls_loss = total_cls_loss
            steps += 1
        # total_baseline_loss /= (len(self.train_loader) * self.args["max_turn"])
        total_reward_loss /= steps
        return total_reward_loss


def split_dial(dial, answer_tokens):
    questions = []
    answers = []
    question = []
    for token in dial:
        if token in answer_tokens:
            answers.append(answer_tokens[token])
            questions.append(question)
            question = []
        else:
            question.append(token)
    return questions, answers
