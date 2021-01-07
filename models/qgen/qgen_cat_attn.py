# -*- coding: utf-8 -*-
import torch.nn as nn
import torch.nn.functional
import torch
from utils.util import MyEmbedding
from models.attention import VisualAttention
from utils.constants import c_len
from models.rnn import GRUEncoder, GRUDecoderBase, GRUDecoderAtt, GRUDecoderCategoryOnce
from icecream import ic


class QGenNetwork(nn.Module):
    def __init__(self, args, tokenizer, device):
        super(QGenNetwork, self).__init__()
        self.args = args
        self.word_embedding_dim = args["embedding_dim"]
        self.device = device
        # encoder decoder共用word embedding
        self.word_embed = MyEmbedding(tokenizer.vocab_list, args["embedding_dim"])
        self.query_encoder = GRUEncoder(
            input_size=self.word_embedding_dim, hidden_size=args["query_hidden"],
            embedding=self.word_embed, n_layers=args["query_layer"],
            p=(0 if args["query_layer"] == 1 else args["query_dropout"]),
            bidirectional=args["query_bi"], out_p=0, device=device
        )
        self.pad_token_id = tokenizer.padding_token
        if args["multi_cate"]:
            ic("mutli-category")
            self.category_embed = nn.Linear(c_len, args["category_dim"])
        else:
            ic("single-category")
            self.category_embed = nn.Embedding(c_len, args["category_dim"])
        torch.nn.init.normal_(self.category_embed.weight, 0.0, 0.1)
        self.ans_embed = nn.Embedding(3, args["answer_dim"])
        torch.nn.init.normal_(self.ans_embed.weight, 0.0, 0.1)
        session_q_dim = args["query_hidden"] + args["answer_dim"] + args["category_dim"]
        image_dim = args["image_dim"] + 76 if args["image_arch"] == "rcnn" else args["image_dim"]
        self.image_compress = nn.Conv1d(
            image_dim, args["v_feature"], 1, 1, padding=0, dilation=1, groups=1, bias=True
        )
        self.visual_dropout = nn.Dropout(args["visual_dropout"])
        if args["visual_att"]:
            ic("with visual attention")
            self.visual_cate_attn = VisualAttention(
                image_dim, args["session_hidden"], args["visual_att_dim"], args["glimpse"])
            self.visual_attn = VisualAttention(
                image_dim, args["session_hidden"]+c_len, args["visual_att_dim"], args["glimpse"])
        else:
            ic("without visual attention")
            self.visual_attn = None
            self.visual_cate_attn = None

        self.session_encoder = GRUEncoder(
            input_size=session_q_dim, hidden_size=args["session_hidden"],
            embedding=None, n_layers=args["session_layer"],
            p=(0 if args["session_layer"] == 1 else args["session_dropout"]),
            bidirectional=False, out_p=0, device=device
        )

        self.cls = "cls" in args["task"]
        if self.cls:
            self.fc2 = nn.Sequential(
                nn.Linear(args["session_hidden"] + args["v_feature"], args["session_hidden"] + args["v_feature"] // 2),
                nn.ReLU(),
                nn.Linear(args["session_hidden"] + args["v_feature"] // 2, c_len)
            )
            self.fc3 = nn.Linear(args["session_hidden"] + 4, args["session_hidden"])
            if args["multi_cate"]:
                self.loss_fn = nn.BCEWithLogitsLoss()
            else:
                self.loss_fn = nn.CrossEntropyLoss()
        self.gen = "gen" in args["task"]
        if self.gen:
            self.decoder_linear = nn.Linear(args["session_hidden"]+args["v_feature"], args["decoder_hidden"])
            if args["decoder_att"]:
                ic("decoder with attention")
                self.decoder = GRUDecoderAtt(
                    n_vocab=tokenizer.no_words, embedding=self.word_embed, embedding_dim=self.word_embedding_dim,
                    category_dim=args["category_dim"], session_dim=args["session_hidden"], v_feature_dim=None,
                    n_layers=args["decoder_layer"], hidden_size=args["decoder_hidden"],
                    attention_size=args["decoder_attention_dim"], dropout=args["decoder_dropout"],
                    beam=args["beam_size"], device=device
                )
            elif "category_once" in args and args["category_once"] is True:
                ic("decoder with category once")
                self.decoder = GRUDecoderCategoryOnce(
                    n_vocab=tokenizer.no_words, embedding=self.word_embed, embedding_dim=self.word_embedding_dim,
                    category_dim=args["category_dim"], n_layers=args["decoder_layer"],
                    hidden_size=args["decoder_hidden"], dropout=args["decoder_dropout"],
                    beam=args["beam_size"], device=device
                )
            else:
                ic("decoder with category each input")
                self.decoder = GRUDecoderBase(
                    n_vocab=tokenizer.no_words, embedding=self.word_embed, embedding_dim=self.word_embedding_dim,
                    category_dim=args["category_dim"], n_layers=args["decoder_layer"],
                    hidden_size=args["decoder_hidden"], dropout=args["decoder_dropout"],
                    beam=args["beam_size"], device=device
                )
            self.gen_loss = nn.CrossEntropyLoss(ignore_index=self.pad_token_id)

    def forward(self, batch):
        device = self.device
        questions, qs_lens, q_indexes, answers, categories, turn, v_feature, bbox, y_cate, y, *_ = batch
        questions, qs_lens, q_indexes = questions.to(device), qs_lens.to(device), q_indexes.to(device)
        answers, categories, turn = answers.to(device), categories.to(device), turn.to(device)
        v_feature, y_cate = v_feature.to(device), y_cate.to(device)
        if self.args["image_arch"] == "vgg16":
            v_feature = v_feature.unsqueeze(1)
        else:
            bbox = bbox.to(self.device)
            v_feature = torch.cat([v_feature, bbox], dim=2)  # B * 36 * (v_dim+bbox)
        v_feature = self.visual_dropout(v_feature)
        outputs, h, mask = self.encoder(questions, qs_lens, categories, answers, q_indexes, turn)
        v_num = v_feature.size(1)
        cls_loss = 0
        category = y_cate
        fc2_out = torch.zeros(category.size(0), c_len, dtype=torch.float, device=self.device)
        if self.cls:
            if self.args["visual_att"]:
                att_scores = self.visual_cate_attn(v_feature, h)
            else:
                att_scores = torch.ones((v_feature.size(0), v_num), device=self.device) / v_num
            v_feature_for_cate = torch.einsum('bn,bnv->bv', [att_scores, v_feature])
            v_feature_for_cate = self.image_compress(v_feature_for_cate.unsqueeze(2)).squeeze(2)
            encoder_cate = torch.cat([h, v_feature_for_cate], dim=-1)
            fc2_out = self.fc2(encoder_cate)
            pred = fc2_out
            if self.args["multi_cate"]:
                cls_loss = self.loss_fn(pred, y_cate)
                pred = (torch.sigmoid(pred) > self.args["th"])
            else:
                cls_loss = self.loss_fn(pred, y_cate.view(-1))
                pred = torch.argmax(pred, dim=1)
            if self.args["no_gt_cate"]:
                category = pred
        if self.args["visual_att"]:
            att_scores = self.visual_attn(v_feature, torch.cat([h, fc2_out], dim=1))
        else:
            att_scores = torch.ones((v_feature.size(0), v_num), device=self.device) / v_num
        v_feature = torch.einsum('bn,bnv->bv', [att_scores, v_feature])
        v_feature = self.image_compress(v_feature.unsqueeze(2)).squeeze(2)
        encoder_output = torch.cat([h, v_feature], dim=-1)
        gen_loss = 0
        if self.gen:
            category = self.category_embed(category)
            decoder_state = self.decoder_linear(encoder_output)
            # decoder & get cross entropy loss
            y = y.to(device)
            if self.args["decoder_att"]:
                pred_y, hidden_out = self.decoder(y[:, :-1], decoder_state, category, None, outputs, mask)
            else:
                pred_y, hidden_out = self.decoder(y[:, :-1], decoder_state, category)
            vocab_size = pred_y.size(2)
            pred_y = pred_y.contiguous().view(-1, vocab_size)
            gt = y[:, 1:].contiguous().view(-1)
            gen_loss = self.gen_loss(pred_y, gt)
        return cls_loss, gen_loss

    def pg_forward(self, questions, qs_lens, q_indexes, answers, categories, turn, v_feature, bbox):
        device = self.device
        questions, qs_lens, q_indexes = questions.to(device), qs_lens.to(device), q_indexes.to(device)
        answers, categories, turn = answers.to(device), categories.to(device), turn.to(device)
        v_feature= v_feature.to(device)

        if self.args["image_arch"] == "vgg16":
            v_feature = v_feature.unsqueeze(1)
        else:
            bbox = bbox.to(self.device)
            v_feature = torch.cat([v_feature, bbox], dim=2)  # B * 36 * (v_dim+bbox)
        v_feature = self.visual_dropout(v_feature)
        outputs, h, mask = self.encoder(questions, qs_lens, categories, answers, q_indexes, turn)
        v_num = v_feature.size(1)
        bsz = v_feature.size(0)

        if self.cls:
            if self.args["visual_att"]:
                att_scores = self.visual_cate_attn(v_feature, h)
            else:
                att_scores = torch.ones((bsz, v_num), device=self.device) / v_num
            v_feature_for_cate = torch.einsum('bn,bnv->bv', [att_scores, v_feature])
            v_feature_for_cate = self.image_compress(v_feature_for_cate.unsqueeze(2)).squeeze(2)
            encoder_cate = torch.cat([h, v_feature_for_cate], dim=-1)
            fc2_out = self.fc2(encoder_cate)
            pred = fc2_out
            if self.args["multi_cate"]:
                labels = (torch.sigmoid(pred) > self.args["th"]).to(torch.float)
            else:
                labels = torch.argmax(pred, dim=1)
        else:
            fc2_out = torch.zeros(bsz, c_len, dtype=torch.float, device=self.device)
            if self.args["multi_cate"]:
                labels = torch.zeros(bsz, c_len, device=self.device, dtype=torch.float)
            else:
                labels = torch.zeros(bsz, device=self.device, dtype=torch.long)

        if self.args["visual_att"]:
            att_scores = self.visual_attn(v_feature, torch.cat([h, fc2_out], dim=1))
        else:
            att_scores = torch.ones((bsz, v_num), device=self.device) / v_num
        v_feature = torch.einsum('bn,bnv->bv', [att_scores, v_feature])
        v_feature = self.image_compress(v_feature.unsqueeze(2)).squeeze(2)
        encoder_output = torch.cat([h, v_feature], dim=-1)

        category = self.category_embed(labels)
        decoder_state = self.decoder_linear(encoder_output)
        if self.args["decoder_att"]:
            vocab_out, hidden_out = self.decoder.greedy_search_generate(
                decoder_state, category)
        else:
            vocab_out, hidden_out = self.decoder.greedy_search_generate(
                decoder_state, category)

        gt = torch.argmax(vocab_out, dim=-1)
        bsz, length, _ = vocab_out.size()
        vocab_out = vocab_out.contiguous().view(bsz*length, -1)
        gt_out = gt.contiguous().view(-1).detach()
        gen_loss = torch.nn.functional.cross_entropy(
            vocab_out, gt_out, reduction="none", ignore_index=self.pad_token_id)
        gen_loss = gen_loss.view(bsz, length)
        # hidden_out = hidden_out.contiguous().view(-1, hidden_out.size(-1)).detach()
        # baseline = self.baseline_linear_2(self.relu(self.baseline_linear_1(hidden_out)))
        # baseline = torch.sigmoid(baseline)
        # baseline = baseline.view(bsz, gt.size(-1))
        baseline = 0
        return baseline, gen_loss, labels, gt

    def pg_forward_with_target(
            self, questions, qs_lens, q_indexes, answers, categories,
            turn, v_feature, bbox, target_cate=None, target=None):
        device = self.device
        questions, qs_lens, q_indexes = questions.to(device), qs_lens.to(device), q_indexes.to(device)
        answers, categories, turn = answers.to(device), categories.to(device), turn.to(device)
        v_feature = v_feature.to(device)

        if self.args["image_arch"] == "vgg16":
            v_feature = v_feature.unsqueeze(1)
        else:
            bbox = bbox.to(self.device)
            v_feature = torch.cat([v_feature, bbox], dim=2)  # B * 36 * (v_dim+bbox)
        v_feature = self.visual_dropout(v_feature)
        outputs, h, mask = self.encoder(questions, qs_lens, categories, answers, q_indexes, turn)
        v_num = v_feature.size(1)
        cls_loss = 0
        bsz = v_feature.size(0)
        if self.cls:
            if self.args["visual_att"]:
                att_scores = self.visual_cate_attn(v_feature, h)
            else:
                att_scores = torch.ones((bsz, v_num), device=self.device) / v_num
            v_feature_for_cate = torch.einsum('bn,bnv->bv', [att_scores, v_feature])
            v_feature_for_cate = self.image_compress(v_feature_for_cate.unsqueeze(2)).squeeze(2)
            encoder_cate = torch.cat([h, v_feature_for_cate], dim=-1)
            fc2_out = self.fc2(encoder_cate)
            pred = fc2_out

            if self.args["multi_cate"]:
                labels = (torch.sigmoid(pred) > self.args["th"]).to(torch.float)
                if target_cate is not None:
                    target_cate = torch.tensor(target_cate, device=self.device, dtype=torch.float)
                    cls_loss = torch.nn.functional.binary_cross_entropy_with_logits(pred, target_cate, reduction="none")
                    labels = target_cate
            else:
                labels = torch.argmax(pred, dim=1)
                if target_cate is not None:
                    target_cate = torch.tensor(target_cate, device=self.device, dtype=torch.long)
                    cls_loss = torch.nn.functional.cross_entropy(
                        pred, target_cate.view(-1), reduction="none")
                    labels = target_cate
        else:
            labels = target_cate
            fc2_out = torch.zeros(bsz, c_len, dtype=torch.float, device=self.device)
        gen_loss = 0

        if target is not None:
            if self.args["visual_att"]:
                att_scores = self.visual_attn(v_feature, torch.cat([h, fc2_out], dim=1))
            else:
                att_scores = torch.ones((bsz, v_num), device=self.device) / v_num
            v_feature = torch.einsum('bn,bnv->bv', [att_scores, v_feature])
            v_feature = self.image_compress(v_feature.unsqueeze(2)).squeeze(2)
            encoder_output = torch.cat([h, v_feature], dim=-1)
            category = self.category_embed(labels)
            decoder_state = self.decoder_linear(encoder_output)

            if self.args["decoder_att"]:
                vocab_out, hidden_out = self.decoder(target[:, :-1], decoder_state, category, None, outputs, mask)
            else:
                vocab_out, hidden_out = self.decoder(target[:, :-1], decoder_state, category)

            bsz, length, _ = vocab_out.size()
            vocab_out = vocab_out.contiguous().view(bsz * length, -1)
            gt_out = target[:, 1:].contiguous().view(-1).detach()
            gen_loss = torch.nn.functional.cross_entropy(
                vocab_out, gt_out, reduction="none", ignore_index=self.pad_token_id)
            gen_loss = gen_loss.view(bsz, length)
        return cls_loss, gen_loss

    def encoder(self, questions, qs_lens, categories, answers, q_indexes, turn):
        # first encoder
        outputs, _ = self.query_encoder(questions, qs_lens)
        # session encoder
        categories = self.category_embed(categories)
        answers = self.ans_embed(answers)
        session_input, session_lengths, session_mask = self.session_rnn_prepare(
            outputs, q_indexes, answers, categories, turn)

        session_outputs, h = self.session_encoder(session_input, session_lengths)  # h: B x hidden
        return session_outputs, h, session_mask

    def session_rnn_prepare(self, outputs, q_index, answers, categories, turns):
        """
        为第二层GRU做数据准备，提取、拼接第一层questions的结果
        :param outputs: B x lens x hidden
        :param q_index: B x max_turn
        :param answers: B x max_turn x ans_dim
        :param categories: B x max_turn x cate_dim
        :param turns: B
        :return session_mask: B x max_turn
        :return session_input: B x max_turn x (hidden + ans_dim + cate_dim)
        :return session_lengths: B
        """
        max_turn = torch.max(turns).item()
        batch_size, _, hidden = outputs.size()
        session_lengths = turns

        session_mask = torch.zeros(batch_size, max_turn)
        inputs = []
        for i in range(batch_size):
            turn = turns[i].item()
            output = outputs[i][q_index[i, :turn]]  # turn x hidden
            answer = answers[i, :turn]  # turn x hidden
            category = categories[i, :turn]  # turn x hidden
            each_input = torch.cat((output, answer, category), dim=-1)  # turn x (hidden + ans_dim + cate_dim)
            turn, hidden_size = each_input.size()
            padding = torch.zeros(max_turn-turn, hidden_size).to(self.device)
            each_input = torch.cat((each_input, padding), dim=0)
            inputs.append(each_input)
            session_mask[i, :turn] = 1.
        session_mask = session_mask.eq(0).to(self.device)
        session_input = torch.stack(inputs)
        return session_input, session_lengths, session_mask

    def generate(self, batch):
        device = self.device
        questions, qs_lens, q_indexes, answers, categories, turn, v_feature, bbox, y_cate, y, *_ = batch
        questions, qs_lens, q_indexes = questions.to(device), qs_lens.to(device), q_indexes.to(device)
        answers, categories, turn = answers.to(device), categories.to(device), turn.to(device)
        v_feature, y_cate = v_feature.to(device), y_cate.to(device)
        # first encoder
        if self.args["image_arch"] == "vgg16":
            v_feature = v_feature.unsqueeze(1)
        else:
            bbox = bbox.to(self.device)
            v_feature = torch.cat([v_feature, bbox], dim=2)  # B * 36 * (v_feature+bbox)
        outputs, h, mask = self.encoder(questions, qs_lens, categories, answers, q_indexes, turn)
        v_num = v_feature.size(1)
        bsz = v_feature.size(0)
        if self.cls:
            if self.args["visual_att"]:
                att_scores = self.visual_cate_attn(v_feature, h)
            else:
                att_scores = torch.ones((bsz, v_num), device=self.device) / v_num
            v_feature_for_cate = torch.einsum('bn,bnv->bv', [att_scores, v_feature])
            v_feature_for_cate = self.image_compress(v_feature_for_cate.unsqueeze(2)).squeeze(2)
            encoder_cate = torch.cat([h, v_feature_for_cate], dim=-1)
            fc2_out = self.fc2(encoder_cate)
            pred = fc2_out
            if self.args["multi_cate"]:
                labels = (torch.sigmoid(pred) > self.args["th"])
            else:

                labels = torch.argmax(pred, dim=1)
        else:
            labels = y_cate
            fc2_out = torch.zeros(bsz, c_len, dtype=torch.float, device=self.device)

        if self.gen:
            # generate
            if self.args["visual_att"]:
                att_scores = self.visual_attn(v_feature, torch.cat([h, fc2_out], dim=1))
            else:
                att_scores = torch.ones((v_feature.size(0), v_num), device=self.device) / v_num
            v_feature = torch.einsum('bn,bnv->bv', [att_scores, v_feature])
            v_feature = self.image_compress(v_feature.unsqueeze(2)).squeeze(2)
            encoder_output = torch.cat([h, v_feature], dim=-1)
            if self.args["multi_cate"]:
                labels = labels.to(torch.float)
            category = self.category_embed(labels)
            decoder_state = self.decoder_linear(encoder_output)
            if self.args["decoder_att"]:
                result = self.decoder.generate(decoder_state, category, None, outputs, mask)
            else:
                result = self.decoder.generate(decoder_state, category)

        else:
            result = y.tolist()
        return result, labels.to(torch.long)


def main():
    from arguments.qgen_args import qgen_arguments
    from data_provider.qgen_dataset import prepare_dataset
    from process_data.tokenizer import GWTokenizer
    parser = qgen_arguments()
    args, _ = parser.parse_known_args()
    args = vars(args)
    print(args)
    tokenizer = GWTokenizer('./../data/dict.json')
    loader = prepare_dataset("./../data/", "test", args, tokenizer)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = QGenNetwork(args, tokenizer, device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), args["lr"])
    data_iter = iter(loader)
    model.train()
    for i in range(5):
        batch = next(data_iter)
        optimizer.zero_grad()
        model.zero_grad()
        cls_loss, gen_loss = model(batch)
        if args["task"] == "cls":
            loss = cls_loss
        elif args["task"] == "gen":
            loss = gen_loss
        else:
            loss = cls_loss + gen_loss
        loss.backward()
        _ = torch.nn.utils.clip_grad_norm_(model.parameters(), args["clip_val"])
        optimizer.step()
        print("loss: {:.4f}".format(loss.item()))
    model.eval()
    batch = next(data_iter)
    result, pred = model.generate(batch)
    print("generate")
    print(pred[0])
    print(tokenizer.decode(result[0]))


if __name__ == '__main__':
    main()
