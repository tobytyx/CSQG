# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from utils.util import MyEmbedding
from models.rnn import GRUEncoder, GRUDecoderBase
from models.attention import VisualAttention, ConcatFusion
from utils.constants import PAD
from icecream import ic


class QGenNetwork(nn.Module):
    def __init__(self, args, tokenizer, device):
        super(QGenNetwork, self).__init__()
        self.args = args
        self.word_embedding_dim = args["embedding_dim"]
        self.device = device
        # encoder decoder共用word embedding
        self.word_embed = MyEmbedding(tokenizer.vocab_list, args["embedding_dim"])
        self.dialogue_encoder = GRUEncoder(
            input_size=self.word_embedding_dim, hidden_size=args["session_hidden"],
            embedding=self.word_embed, n_layers=args["session_layer"],
            p=(0 if args["session_layer"] == 1 else args["session_dropout"]),
            bidirectional=args["session_bi"], out_p=0, device=device
        )
        session_v_dim = args["image_dim"] + 76 if args["image_arch"] == "rcnn" else args["image_dim"]
        self.image_compress = nn.Conv1d(
            args["image_dim"], args["v_feature"], 1, 1, padding=0, dilation=1, groups=1, bias=True
        )
        torch.nn.init.kaiming_normal_(self.image_compress.weight, nonlinearity='relu')
        torch.nn.init.constant_(self.image_compress.bias, 0.0)
        if args["visual_att"]:
            self.attention = VisualAttention(
                session_v_dim, args["session_hidden"], args["visual_att_dim"], args["glimpse"])
            self.glimpse = args["glimpse"]
        else:
            self.attention = None
        self.visual_dropout = nn.Dropout(args["visual_dropout"])
        self.fusion = ConcatFusion(args["session_hidden"], args["v_feature"], args["decoder_hidden"])
        self.decoder = GRUDecoderBase(
            n_vocab=tokenizer.no_words, embedding=self.word_embed, embedding_dim=self.word_embedding_dim,
            category_dim=None, n_layers=args["decoder_layer"], hidden_size=args["decoder_hidden"],
            dropout=args["decoder_dropout"], beam=args["beam_size"], device=device
        )
        self.gen_loss = nn.CrossEntropyLoss(ignore_index=PAD)

    def forward(self, batch):
        device = self.device
        dialogues, dial_lens, a_indexes, turns, v_feature, bbox, y, *_ = batch
        dialogues, dial_lens, a_indexes = dialogues.to(device), dial_lens.to(device), a_indexes.to(device)
        turns, v_feature, bbox, y = turns.to(device), v_feature.to(device), bbox.to(device), y.to(device)

        decoder_state = self.encoder(dialogues, dial_lens, v_feature, bbox)
        pred_y = self.decoder(y[:, :-1], decoder_state)

        pred_y = pred_y.contiguous().view(-1, pred_y.size(2))
        gt = y[:, 1:].contiguous().view(-1)
        gen_loss = self.gen_loss(pred_y, gt)
        return 0, gen_loss

    def encoder(self, dialogues, dial_lens, v_feature, bbox):
        outputs, h = self.dialogue_encoder(dialogues, dial_lens)
        if self.args["image_arch"] == "vgg16":
            v_feature = v_feature.unsqueeze(1)
        else:
            v_feature = torch.cat([v_feature, bbox], dim=2)  # B * l * (v_dim+bbox)
        if self.attention is not None:
            visual_score = self.attention(v_feature, h)  # B x l
        else:
            visual_score = torch.ones(
                v_feature.size(0), v_feature.size(1), device=self.device) / v_feature.size(1)
        v_feature = torch.einsum("blh,bl->bh", [v_feature, visual_score])
        v_feature = v_feature.unsqueeze(-1)
        v_feature = self.image_compress(v_feature)
        v_feature = v_feature.squeeze(-1)
        v_feature = self.visual_dropout(v_feature)
        decoder_state = self.fusion(h, v_feature)
        return decoder_state

    def generate(self, batch):
        device = self.device
        dialogues, dial_lens, a_indexes, turns, v_feature, bbox, *_ = batch
        dialogues, dial_lens, a_indexes = dialogues.to(device), dial_lens.to(device), a_indexes.to(device)
        turns, v_feature, bbox = turns.to(device), v_feature.to(device), bbox.to(device)

        decoder_state = self.encoder(dialogues, dial_lens, v_feature, bbox)
        result = self.decoder.generate(decoder_state)
        return result, None


def main():
    from arguments.qgen_args import qgen_arguments
    from data_provider.qgen_baseline_dataset import prepare_dataset
    from process_data.tokenizer import GWTokenizer
    parser = qgen_arguments()
    args, _ = parser.parse_known_args()
    args = vars(args)
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
        _, loss = model(batch)
        loss.backward()
        _ = torch.nn.utils.clip_grad_norm_(model.parameters(), args["clip_val"])
        optimizer.step()
        print("loss: {:.4f}".format(loss.item()))
    model.eval()
    batch = next(data_iter)
    result, _ = model.generate(batch)
    print("generate")
    print(tokenizer.decode(result[0]))


if __name__ == '__main__':
    main()
