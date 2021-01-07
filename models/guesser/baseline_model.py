# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional
from models.rnn import GRUEncoder


class GuesserNetwork(nn.Module):
    def __init__(self, opt, tokenizer, device):
        super(GuesserNetwork, self).__init__()
        word_num = tokenizer.no_words
        self.opt = opt
        word_embedding_dim = opt["embedding_dim"]
        category_num = opt["n_category"]
        category_dim = opt["category_embed_dim"]
        hidden_size = opt["hidden"]
        n_layers = opt["layer"]
        dropout = opt["dropout"]
        image_dim = opt["image_dim"]
        self.cell = GRUEncoder(input_size=word_embedding_dim, hidden_size=hidden_size, embedding=None,
                               n_layers=n_layers, p=dropout, bidirectional=True, out_p=dropout, device=device)
        self.word_embedding = nn.Embedding(word_num, word_embedding_dim)
        torch.nn.init.normal_(self.word_embedding.weight, 0.0, 0.1)
        self.category_embedding = nn.Embedding(category_num + 1, category_dim)
        torch.nn.init.normal_(self.category_embedding.weight, 0.0, 0.1)
        obj_dim = opt["MLP1_hidden"]
        self.final_dim = opt["MLP2_hidden"]

        if opt["image"]:
            self.linear = nn.Linear(opt["hidden"]+image_dim, self.final_dim)

        self.mlp = nn.Sequential(
            nn.Linear(category_dim+8, obj_dim),
            nn.ReLU(),
            nn.Linear(obj_dim, self.final_dim),
            nn.ReLU()
        )
        self.device = device

    def forward(self, batch):
        dialog, dialog_lengths, img, obj_mask, cats, spas, *_ = batch
        dialog, dialog_lengths, img = dialog.to(self.device), dialog_lengths.to(self.device), img.to(self.device)
        obj_mask, cats, spas = obj_mask.to(self.device), cats.to(self.device), spas.to(self.device)
        dialog = self.word_embedding(dialog)
        cats = self.category_embedding(cats)  # B*N*256
        _, hidden = self.cell(dialog, dialog_lengths)
        if self.opt["image"]:
            x = torch.cat((img, hidden), dim=-1)
            x = self.linear(x)
            hidden = torch.tanh(x)
        obj = torch.cat((cats, spas), dim=2)  # B*N*D
        obj = self.mlp(obj)
        predict = torch.einsum('bh,bnh->bn', [hidden, obj])
        obj_mask = obj_mask.eq(0)
        predict = predict.masked_fill(obj_mask, -float('inf')).type_as(predict)
        return predict


def main():
    from arguments.guesser_args import guesser_arguments
    from data_provider.guesser_dataset import prepare_dataset
    from process_data.tokenizer import GWTokenizer
    from utils.calculate_util import calculate_accuracy
    parser = guesser_arguments()
    args, _ = parser.parse_known_args()
    args = vars(args)
    tokenizer = GWTokenizer('./../data/dict.json')
    loader = prepare_dataset("./../data/", "test", args, tokenizer)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GuesserNetwork(args, tokenizer, device).to(device)
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
    _, predicted = torch.max(output, dim=1)
    predicted = predicted.tolist()
    target = target.tolist()
    for i in range(len(predicted)):
        print("{}, {}".format(predicted[i], target[i]))
    print("acc: {:4f}".format(accuracy))


if __name__ == '__main__':
    main()
