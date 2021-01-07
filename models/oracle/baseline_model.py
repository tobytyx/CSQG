# -*- coding: utf8 -*-
import torch
import torch.nn as nn
import torch.nn.functional
from models.rnn import GRUEncoder
num_classes = 3


class OracleNetwork(nn.Module):
    def __init__(self, opt, tokenizer, device):
        super(OracleNetwork, self).__init__()
        self.word_num = tokenizer.no_words
        self.word_embedding_dim = opt["embedding_dim"]
        self.category_num = opt["n_category"]
        self.category_embedding_dim = opt["category_embed_dim"]
        self.hidden_size = opt["hidden"]
        self.n_layers = opt["layer"]
        self.dropout = opt["dropout"]
        self.opt = opt
        self.rnn = GRUEncoder(
            input_size=self.word_embedding_dim, hidden_size=self.hidden_size, embedding=None,
            n_layers=self.n_layers, p=self.dropout, bidirectional=opt["bidirectional"],
            out_p=self.dropout, device=device
        )
        self.word_embedding = nn.Embedding(self.word_num, self.word_embedding_dim)
        self.category_embedding = nn.Embedding(self.category_num+1, self.category_embedding_dim)
        torch.nn.init.normal_(self.word_embedding.weight, 0.0, 0.1)
        torch.nn.init.normal_(self.category_embedding.weight, 0.0, 0.1)
        fc_dim = self.hidden_size
        if opt["category"]:
            fc_dim += self.category_embedding_dim
        if opt["spatial"]:
            fc_dim += 8  # 空间位置维度信息
        if opt["image"]:
            fc_dim += opt["image_dim"]
        if opt["crop"]:
            fc_dim += opt["crop_dim"]
        self.fc1 = nn.Linear(fc_dim, opt["MLP_hidden"])
        self.fc2 = nn.Linear(opt["MLP_hidden"], num_classes)
        self.device = device

    def forward(self, batch):
        questions, q_lens, imgs, crops, cats, spas, *_ = batch
        questions, q_lens, imgs = questions.to(self.device), q_lens.to(self.device), imgs.to(self.device)
        crops, cats, spas = crops.to(self.device), cats.to(self.device), spas.to(self.device)
        questions = self.word_embedding(questions)
        _, hidden = self.rnn(questions, q_lens)

        if self.opt["category"]:
            cate = self.category_embedding(cats)
            hidden = torch.cat((hidden, cate), dim=1)
        if self.opt["spatial"]:
            hidden = torch.cat((hidden, spas), dim=1)
        if self.opt["crop"]:
            hidden = torch.cat((hidden, crops), dim=1)
        if self.opt["image"]:
            hidden = torch.cat((hidden, imgs), dim=1)
        hidden = torch.nn.functional.relu(self.fc1(hidden))
        y = self.fc2(hidden)
        return y


def main():
    from arguments.oracle_args import oracle_arguments
    from data_provider.oracle_dataset import prepare_dataset
    from process_data.tokenizer import GWTokenizer
    from utils.calculate_util import calculate_accuracy
    parser = oracle_arguments()
    args, _ = parser.parse_known_args()
    args = vars(args)
    tokenizer = GWTokenizer('./../data/dict.json')
    loader = prepare_dataset("./../data/", "test", args, tokenizer)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = OracleNetwork(args, tokenizer, device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), args["lr"])
    data_iter = iter(loader)
    model.train()
    for i in range(20):
        batch = next(data_iter)
        optimizer.zero_grad()
        model.zero_grad()
        output = model(batch)
        target = batch[-1].to(device).long()  # target object index
        loss = torch.nn.functional.cross_entropy(output, target)
        loss.backward()
        _ = torch.nn.utils.clip_grad_norm_(model.parameters(), args["clip_val"])
        optimizer.step()
        print("loss: {:.4f}".format(loss.item()))
    model.eval()
    batch = next(data_iter)
    output = model(batch)
    target = batch[-1].to(device).long()
    _, accuracy = calculate_accuracy(output, target)
    print("acc: {:4f}".format(accuracy))


if __name__ == '__main__':
    main()
