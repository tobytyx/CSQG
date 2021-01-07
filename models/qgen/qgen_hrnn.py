# -*- coding: utf-8 -*-
import torch.nn as nn
import torch
from utils.util import MyEmbedding
from models.attention import VisualAttention
from utils import constants
from models.rnn import GRUEncoder, GRUDecoderBase


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
        self.ans_embed = nn.Embedding(3, args["answer_dim"])
        torch.nn.init.normal_(self.ans_embed.weight, 0.0, 0.1)
        session_q_dim = args["query_hidden"] + args["answer_dim"]
        image_dim = args["image_dim"] + 76 if args["image_arch"] == "rcnn" else args["image_dim"]
        self.image_compress = nn.Conv1d(
            image_dim, args["v_feature"], 1, 1, padding=0, dilation=1, groups=1, bias=True
        )
        self.visual_dropout = nn.Dropout(args["visual_dropout"])
        if args["visual_att"]:
            self.visual_attn = VisualAttention(
                image_dim, args["session_hidden"], args["visual_att_dim"], args["glimpse"])
        else:
            self.visual_attn = None

        self.session_encoder = GRUEncoder(
            input_size=session_q_dim, hidden_size=args["session_hidden"],
            embedding=None, n_layers=args["session_layer"],
            p=(0 if args["session_layer"] == 1 else args["session_dropout"]),
            bidirectional=False, out_p=0, device=device
        )

        self.decoder_linear = nn.Linear(args["session_hidden"]+args["v_feature"], args["decoder_hidden"])
        self.decoder = GRUDecoderBase(
            n_vocab=tokenizer.no_words, embedding=self.word_embed, embedding_dim=self.word_embedding_dim,
            category_dim=None, n_layers=args["decoder_layer"], hidden_size=args["decoder_hidden"],
            dropout=args["decoder_dropout"], beam=args["beam_size"], device=device
        )
        self.gen_loss = nn.CrossEntropyLoss(ignore_index=constants.PAD)

    def forward(self, batch):
        device = self.device
        questions, qs_lens, q_indexes, answers, _, turn, v_feature, bbox, _, y, *_ = batch
        questions, qs_lens, q_indexes = questions.to(device), qs_lens.to(device), q_indexes.to(device)
        answers, turn = answers.to(device), turn.to(device)
        v_feature = v_feature.to(device)
        if self.args["image_arch"] == "vgg16":
            v_feature = v_feature.unsqueeze(1)
        else:
            bbox = bbox.to(self.device)
            v_feature = torch.cat([v_feature, bbox], dim=2)  # B * 36 * (v_dim+bbox)
        v_feature = self.visual_dropout(v_feature)
        outputs, h, mask = self.encoder(questions, qs_lens, answers, q_indexes, turn)
        v_num = v_feature.size(1)
        if self.args["visual_att"]:
            att_scores = self.visual_attn(v_feature, h)
        else:
            att_scores = torch.ones((v_feature.size(0), v_num), device=self.device) / v_num
        v_feature = torch.einsum('bn,bnv->bv', [att_scores, v_feature])
        v_feature = self.image_compress(v_feature.unsqueeze(2)).squeeze(2)
        decoder_state = self.decoder_linear(torch.cat([h, v_feature], dim=-1))
        # decoder & get cross entropy loss
        y = y.to(device)
        pred_y = self.decoder(y[:, :-1], decoder_state)
        vocab_size = pred_y.size(2)
        pred_y = pred_y.contiguous().view(-1, vocab_size)
        gt = y[:, 1:].contiguous().view(-1)
        gen_loss = self.gen_loss(pred_y, gt)
        return 0, gen_loss

    def encoder(self, questions, qs_lens, answers, q_indexes, turn):
        # first encoder
        outputs, _ = self.query_encoder(questions, qs_lens)
        # session encoder
        answers = self.ans_embed(answers)
        session_input, session_lengths, session_mask = self.session_rnn_prepare(
            outputs, q_indexes, answers, turn)

        session_outputs, h = self.session_encoder(session_input, session_lengths)  # h: B x hidden
        return session_outputs, h, session_mask

    def session_rnn_prepare(self, outputs, q_index, answers, turns):
        """
        为第二层GRU做数据准备，提取、拼接第一层questions的结果
        :param outputs: B x lens x hidden
        :param q_index: B x max_turn
        :param answers: B x max_turn x ans_dim
        :param turns: B
        :return session_mask: B x max_turn
        :return session_input: B x max_turn x (hidden + ans_dim)
        :return session_lengths: B
        """
        max_turn = torch.max(turns).item()
        batch_size, _, hidden = outputs.size()
        session_lengths = turns

        session_mask = torch.zeros(batch_size, max_turn)
        inputs = []
        for i in range(batch_size):
            turn = turns[i].item()
            output = outputs[i][q_index[i][:turn]]  # turn x hidden
            answer = answers[i][:turn]  # turn x hidden
            each_input = torch.cat((output, answer), dim=-1)  # turn x (hidden + ans_dim)
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
        questions, qs_lens, q_indexes, answers, _, turn, v_feature, bbox, _, y, *_ = batch
        questions, qs_lens, q_indexes = questions.to(device), qs_lens.to(device), q_indexes.to(device)
        answers, turn = answers.to(device), turn.to(device)
        v_feature = v_feature.to(device)
        # first encoder
        if self.args["image_arch"] == "vgg16":
            v_feature = v_feature.unsqueeze(1)
        else:
            bbox = bbox.to(self.device)
            v_feature = torch.cat([v_feature, bbox], dim=2)  # B * 36 * (v_feature+bbox)
        outputs, h, mask = self.encoder(questions, qs_lens, answers, q_indexes, turn)
        v_num = v_feature.size(1)
        if self.args["visual_att"]:
            att_scores = self.visual_attn(v_feature, h)
        else:
            att_scores = torch.ones((v_feature.size(0), v_num), device=self.device) / v_num
        v_feature = torch.einsum('bn,bnv->bv', [att_scores, v_feature])
        v_feature = self.image_compress(v_feature.unsqueeze(2)).squeeze(2)
        decoder_state = self.decoder_linear(torch.cat([h, v_feature], dim=-1))
        result = self.decoder.generate(decoder_state)
        return result, None
